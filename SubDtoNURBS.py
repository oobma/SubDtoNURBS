bl_info = {
    "name": "Limit Surface to NURBS (v0.2.1-Repair flat patches detection)",
    "author": "oobma, Gemini (Final Architect) DeepSeek, Qwen (Consultants)",
    "version": (0, 2, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > SubDtoNURBS",
    "description": "Convert subdivision surfaces to NURBS",
    "category": "Object",
}

import bpy
import bmesh
from mathutils import Vector
import time
import numpy as np
import heapq
import os
from pathlib import Path

# --- Helper Functions ---
def conjugate_gradient_solver(A, b, max_iterations=1000, tolerance=1e-5):
    n=len(b);x=np.zeros(n);r=b-(A@x);p=r.copy();rs_old=np.dot(r,r)
    if np.sqrt(rs_old)<tolerance:return x
    for i in range(max_iterations):
        Ap=A@p;alpha=rs_old/np.dot(p,Ap);x+=alpha*p;r-=alpha*Ap;rs_new=np.dot(r,r)
        if np.sqrt(rs_new)<tolerance:break
        p=r+(rs_new/rs_old)*p;rs_old=rs_new
    else:print(f"  Warning: CG did not converge.")
    return x
def find_edge_between_verts(v1,v2):
    for e in v1.link_edges:
        if e.other_vert(v1)==v2:return e
    return None
def is_patch_truly_flat(patch, tolerance=1e-4):
    if not patch: return False
    p00, p03, p30 = patch[0][0], patch[0][3], patch[3][0]
    plane_co = p00; plane_no = (p30 - p00).cross(p03 - p00)
    if plane_no.length_squared < 1e-12: return False
    plane_no.normalize()
    for r in range(4):
        for c in range(4):
            dist = (patch[r][c] - plane_co).dot(plane_no)
            if abs(dist) > tolerance: return False
    return True
def append_object_from_surfacepsycho_assets(object_name: str):
    try:
        extensions_path = Path(bpy.utils.resource_path('USER')) / "extensions" / "user_default" / "surfacepsycho" / "assets" / "assets.blend"
        if not extensions_path.exists(): return None
        with bpy.data.libraries.load(str(extensions_path), link=False) as (data_from, data_to):
            if object_name in data_from.objects: data_to.objects = [object_name]
            else: return None
        return data_to.objects[0] if data_to.objects else None
    except Exception: return None
def set_first_vertex_smooth(target_obj):
    if target_obj and target_obj.type == 'MESH' and target_obj.data:
        mesh_data = target_obj.data
        bm = bmesh.new();bm.from_mesh(mesh_data);bm.verts.ensure_lookup_table()
        if len(bm.verts) > 0:
            for face in bm.verts[0].link_faces: face.smooth = True
        bm.to_mesh(mesh_data);bm.free();mesh_data.update()

# --- Settings and Algorithm Classes ---
class NurbsConverterSettings(bpy.types.PropertyGroup):
    output_type: bpy.props.EnumProperty(name="Output Type",description="Choose output type.",items=[('BLENDER_NURBS', "Blender NURBS", "" ),('PSYCHOPATCH', "PsychoPatch", "")],default='BLENDER_NURBS')
    join_patches: bpy.props.BoolProperty(name="Join Patches", description="Join all patches into a single object.", default=True)
    add_weld_modifier: bpy.props.BoolProperty(name="Add Weld", description="Add a Weld modifier to close micro-seams.", default=True)
    add_subd_modifier: bpy.props.BoolProperty(name="Add Subdivision", description="Add a Subdivision Surface modifier to refine the mesh.", default=True)
    add_smooth_modifier: bpy.props.BoolProperty(name="Add Smooth", description="Add a Smooth modifier for perfect shading.", default=True)
    generate_ruled_surfaces:bpy.props.BoolProperty(name="Correct Ruled Surfaces",description="Smooth transition between opposite folded edges with G1 continuity.",default=True)
    force_planar_faces:bpy.props.BoolProperty(name="Force Planar Faces",description="Post-process faces with creased boundaries to enforce perfect flatness.",default=True)
    planar_tolerance:bpy.props.FloatProperty(name="Planar Tolerance",description="Sensitivity for detecting a face as planar.",default=1e-4,min=1e-7,max=1e-2,subtype='FACTOR',precision=6)
    g1_transition_weight:bpy.props.FloatProperty(name="G1 Transition Weight",description="Weaken G1 between flat/curved patches to prevent bulging.",default=0.05,min=0.0,max=1.0,subtype='FACTOR')
    exceptional_weight:bpy.props.FloatProperty(name="Exceptional Weight",description="Control G1 strength at exceptional vertices to reduce denting.",default=0.1,min=0.0,max=1.0,subtype='FACTOR')
    fairing_weight:bpy.props.FloatProperty(name="Fairing Weight",description="Pull CPs towards their 'ideal' shape to stabilize the solution.",default=0.01,min=0.0,max=1.0,subtype='FACTOR')

class GeometryAnalyzer:
    def __init__(self,bm,w):self.bm,self.w=bm,w;self.ev={v.index for v in self.bm.verts if len(v.link_edges)!=4 and not v.is_boundary}
    def get_g1_constraint_weight(self,v_idx):return self.w if v_idx in self.ev else 1.0

class EdgeCentricStitcher:
    def __init__(self,bm,cache,obj,settings,planar_map,ruled_map,crease_layer):
        self.bm,self.cache,self.obj,self.settings=bm,cache,obj,settings;self.planar_map=planar_map;self.ruled_map=ruled_map;self.crease_layer=crease_layer;self.initial={i:[r[:] for r in v['cps']] for i,v in self.cache.items()};self.final={i:[r[:] for r in v['cps']] for i,v in self.cache.items()};self.map={e.index:[] for e in self.bm.edges};self.geo=GeometryAnalyzer(self.bm,self.settings.exceptional_weight)
    def run(self):
        self._build_map();deps=bpy.context.evaluated_depsgraph_get();obj_eval=self.obj.evaluated_get(deps);self.verts=[v.co.copy() for v in obj_eval.data.vertices];self._enforce_G0();self._solve_G1();return self.final
    def _get_side_map(self,f,o):
        v0,v1,v2,v3=(l.vert for l in f.loops);et,er,eb,el=find_edge_between_verts(v0,v1),find_edge_between_verts(v1,v2),find_edge_between_verts(v2,v3),find_edge_between_verts(v3,v0)
        return {'U0':et,'U1':eb,'V0':el,'V1':er} if o==0 else {'U0':el,'U1':er,'V0':et,'V1':eb}
    def _build_map(self):
        for f in self.bm.faces:
            if f.index not in self.cache:continue
            o=self.cache[f.index]['orientation'];sm=self._get_side_map(f,o)
            for s,e in sm.items():
                if e:self.map[e.index].append((f.index,s))
    def _enforce_G0(self):
        for e in self.bm.edges:
            if e.is_boundary or len(self.map.get(e.index,[]))!=2:continue
            (fa_idx,sa),(fb_idx,sb)=self.map[e.index];g0_boundary=None;is_creased=e[self.crease_layer]>0.99
            if is_creased:
                fa_is_special = self.planar_map.get(fa_idx, False) or self.ruled_map.get(fa_idx, False)
                fb_is_special = self.planar_map.get(fb_idx, False) or self.ruled_map.get(fb_idx, False)
                if fa_is_special and not fb_is_special: g0_boundary = self._get_b(self.initial[fa_idx], sa)
                elif fb_is_special and not fa_is_special: g0_boundary = self._get_b(self.initial[fb_idx], sb, reverse=True)
            if g0_boundary is None: 
                pa,pb=self.initial[fa_idx],self.initial[fb_idx];ca,cb=self._get_b(pa,sa),self._get_b(pb,sb,reverse=True)
                g0_boundary=[(p1+p2)/2.0 for p1,p2 in zip(ca,cb)]
            vs,ve=e.verts;cos,coe=self.verts[vs.index],self.verts[ve.index]
            if(g0_boundary[0]-cos).length_squared>(g0_boundary[0]-coe).length_squared:cos,coe=coe,cos
            g0_boundary[0],g0_boundary[3]=cos,coe
            self._set_b(self.final[fa_idx],sa,g0_boundary);self._set_b(self.final[fb_idx],sb,g0_boundary,reverse=True)
    def _solve_G1(self):
        v={(f,r,c):i for i,(f,r,c) in enumerate((fi,r,c) for fi in self.initial for r in range(1,3) for c in range(1,3))}
        if not v:return
        rs,bs=[],[]
        for e in self.bm.edges:
            if e.is_boundary or len(self.map.get(e.index,[]))!=2:continue
            (fa,sa),(fb,sb)=self.map[e.index];ca,cb,bd=self._get_i(sa),self._get_i(sb,reverse=True),self._get_bi(sa)
            vs,ve=e.verts;is_transition=self.planar_map.get(fa,False)!=self.planar_map.get(fb,False)
            for i in range(4):
                w=1.0
                if is_transition:w*=self.settings.g1_transition_weight
                if i==0:w*=self.geo.get_g1_constraint_weight(vs.index)
                elif i==3:w*=self.geo.get_g1_constraint_weight(ve.index)
                rb,cb_idx=bd[i];b_val=w*(2.0*self.final[fa][rb][cb_idx]);r=[]
                ar,ac=ca[i]
                if(fa,ar,ac) in v:r.append((v[(fa,ar,ac)],w))
                else:b_val-=w*self.final[fa][ar][ac]
                br,bc=cb[i]
                if(fb,br,bc) in v:r.append((v[(fb,br,bc)],w))
                else:b_val-=w*self.final[fb][br][bc]
                if r:rs.append(r);bs.append(b_val)
        wf=self.settings.fairing_weight
        for(f,r,c),i in v.items():rs.append([(i,wf)]);bs.append(wf*self.initial[f][r][c])
        n=len(v);A=np.zeros((n,n));bnp=np.array([[vec.x,vec.y,vec.z] for vec in bs]);Atb=np.zeros((n,3))
        for i,r in enumerate(rs):
            for i1,v1 in r:
                Atb[i1]+=v1*bnp[i]
                for i2,v2 in r:A[i1,i2]+=v1*v2
        sx,sy,sz=conjugate_gradient_solver(A,Atb[:,0]),conjugate_gradient_solver(A,Atb[:,1]),conjugate_gradient_solver(A,Atb[:,2])
        for(f,r,c),i in v.items():self.final[f][r][c]=Vector((sx[i],sy[i],sz[i]))
    def _get_b(self,p,s,reverse=False):
        if s=='U0':r=[p[0][i].copy() for i in range(4)]
        elif s=='U1':r=[p[3][3-i].copy() for i in range(4)]
        elif s=='V0':r=[p[3-i][0].copy() for i in range(4)]
        else:r=[p[i][3].copy() for i in range(4)]
        return r[::-1] if reverse else r
    def _set_b(self,p,s,c,reverse=False):
        cp=c[::-1] if reverse else c[:];
        if s=='U0':
            for i in range(4):p[0][i]=cp[i]
        elif s=='U1':
            for i in range(4):p[3][3-i]=cp[i]
        elif s=='V0':
            for i in range(4):p[3-i][0]=cp[i]
        elif s=='V1':
            for i in range(4):p[i][3]=cp[i]
    def _get_bi(self,s,reverse=False):
        if s=='U0':r=[(0,i) for i in range(4)]
        elif s=='U1':r=[(3,3-i) for i in range(4)]
        elif s=='V0':r=[(3-i,0) for i in range(4)]
        else:r=[(i,3) for i in range(4)]
        return r[::-1] if reverse else r
    def _get_i(self,s,reverse=False):
        if s=='U0':r=[(1,i) for i in range(4)]
        elif s=='U1':r=[(2,3-i) for i in range(4)]
        elif s=='V0':r=[(3-i,1) for i in range(4)]
        else:r=[(i,2) for i in range(4)]
        return r[::-1] if reverse else r

class LimitSurfaceFitter:
    def __init__(self,context,obj,settings):
        self.context,self.obj,self.matrix_world=context,obj,obj.matrix_world.copy();self.settings=settings;self.template_data=self.create_nurbs_template();self.report=lambda*args,**kwargs:None
        # <<< BUGFIX FINAL: Inicializar los atributos de la clase >>>
        self.oriented_fit_cache = {}; self.ruled_map = {}; self.planar_map = {}
        
    def run_conversion(self):
        start_time=time.time();print(f"Starting NURBS conversion ({bl_info['name']})...")
        bm_orig=bmesh.new();bm_orig.from_mesh(self.obj.data);bm_eval=self.get_evaluated_bmesh()
        crease_layer = bm_orig.edges.layers.float.get("crease_edge") or bm_orig.edges.layers.float.new("crease_edge")
        bm_orig.verts.ensure_lookup_table();bm_orig.faces.ensure_lookup_table();bm_orig.edges.ensure_lookup_table()
        orig_vert_to_eval_map={ov.index:ev for ov,ev in zip(bm_orig.verts,bm_eval.verts)}
        print("Phase 1: Global initial setting...")
        self.oriented_fit_cache = {}; self.ruled_map = {}
        for f in bm_orig.faces:
            if len(f.verts) != 4: continue
            is_ruled = False
            if self.settings.generate_ruled_surfaces:
                is_ruled = self.check_if_ruled_face(f, crease_layer)
            if is_ruled:
                 # Esta lógica podría ser más compleja en el futuro si se admiten caras regladas no planas
                pass # Por ahora, dejamos que el trazador normal lo maneje
            trace_result = self.trace_face_grid(f, orig_vert_to_eval_map, bm_eval)
            if not trace_result: continue
            point_grid, orientation, m, n = trace_result
            control_points = self.fit_cps_to_grid(point_grid, m, n)
            if not control_points: continue
            self.oriented_fit_cache[f.index] = {'cps': control_points, 'orientation': orientation}
            if is_ruled: self.ruled_map[f.index] = True
        if not self.oriented_fit_cache:
            self.report({'ERROR'},"Initial adjustment failure.");bm_orig.free();bm_eval.free();return None
        self.planar_map = {idx: is_patch_truly_flat(data['cps'], self.settings.planar_tolerance) for idx, data in self.oriented_fit_cache.items() if idx not in self.ruled_map}
        print("Phase 2: Global sewing...")
        stitcher=EdgeCentricStitcher(bm_orig,self.oriented_fit_cache,self.obj,self.settings, self.planar_map, self.ruled_map, crease_layer)
        final_patches=stitcher.run()
        if self.settings.force_planar_faces:
            print("Phase 2.5: Correction for design intent...")
            final_patches=self.get_planarized_patches(final_patches)
        print(f"Phase 3: Creating objects type '{self.settings.output_type}'...")
        collection = None
        if self.settings.output_type == 'BLENDER_NURBS':
            collection = self._create_nurbs_objects(final_patches, f"{self.obj.name}_NURBS_Patches")
        elif self.settings.output_type == 'PSYCHOPATCH':
            collection = self._create_psychopatch_objects(final_patches, f"{self.obj.name}_Psycho_Patches")
            if not collection: self.report({'ERROR'}, "Unable to load ‘NURBS Patch’ from surfacepsycho.")
        bm_orig.free();bm_eval.free();print(f"✅ NURBS conversion completed in {time.time()-start_time:.2f}s.")
        return collection
    def check_if_ruled_face(self, face, crease_layer):
        loops = list(face.loops); edges = [l.edge for l in loops]; creases = [e[crease_layer] for e in edges]
        if creases[0] > 0.99 and creases[2] > 0.99 and creases[1] < 0.01 and creases[3] < 0.01: return True
        if creases[1] > 0.99 and creases[3] > 0.99 and creases[0] < 0.01 and creases[2] < 0.01: return True
        return False
    def get_planarized_patches(self, stitched_patches):
        print("   - Applying post-processing to force planarity...")
        new_patches = {}; planar_faces_count = 0
        for face_idx, final_cps in stitched_patches.items():
            new_control_points = [[cp.copy() for cp in row] for row in final_cps]
            if self.planar_map.get(face_idx, False):
                planar_faces_count += 1
                p00, p03, p30 = final_cps[0][0], final_cps[0][3], final_cps[3][0]
                plane_co = p00; plane_no = (p30 - p00).cross(p03 - p00)
                if plane_no.length_squared > 1e-12:
                    plane_no.normalize()
                    for r in range(4):
                        for c in range(4):
                            dist = (new_control_points[r][c] - plane_co).dot(plane_no)
                            new_control_points[r][c] -= dist * plane_no
            new_patches[face_idx] = new_control_points
        print(f"     - Flatness forcing applied to {planar_faces_count} patches.")
        return new_patches
    def _create_nurbs_objects(self,patches, col_name):
        nurbs_col=bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
        if col_name not in self.context.scene.collection.children:self.context.scene.collection.children.link(nurbs_col)
        for obj in nurbs_col.objects:bpy.data.objects.remove(obj,do_unlink=True)
        created_objects = []
        for face_idx,cps in patches.items():
            name,nurbs_data=f"{self.obj.name}_Patch_{face_idx:04d}",self.template_data.copy();spline=nurbs_data.splines[0]
            flat_cps=[cp for row in reversed(cps) for cp in row]
            for i,pt in enumerate(spline.points):pt.co=(*(self.matrix_world@flat_cps[i]),1.0)
            nurbs_obj=bpy.data.objects.new(name,nurbs_data);nurbs_col.objects.link(nurbs_obj)
            created_objects.append(nurbs_obj)
        if self.settings.join_patches and created_objects:
            print("   - Joining patches and adding modifiers...")
            base_obj = created_objects[0]
            override_context = self.context.copy();override_context['active_object'] = base_obj;override_context['selected_editable_objects'] = created_objects
            with bpy.context.temp_override(**override_context): bpy.ops.object.join()
            joined_obj = base_obj
            if joined_obj and joined_obj.name in bpy.data.objects:
                joined_obj.name = f"{self.obj.name}_Joined_NURBS"
                if self.settings.add_weld_modifier: joined_obj.modifiers.new(name="Weld", type='WELD')
                if self.settings.add_subd_modifier: joined_obj.modifiers.new(name="Subdivision", type='SUBSURF')
                if self.settings.add_smooth_modifier: joined_obj.modifiers.new(name="Smooth", type='SMOOTH')
                if self.settings.add_weld_modifier: joined_obj.modifiers["Weld"].merge_threshold = 0.0001
                if self.settings.add_subd_modifier: joined_obj.modifiers["Subdivision"].levels = 1
                if self.settings.add_smooth_modifier: joined_obj.modifiers["Smooth"].iterations = 4
        return nurbs_col
    def _create_psychopatch_objects(self, patches, col_name):
        template_obj = bpy.data.objects.get("NURBS Patch") or append_object_from_surfacepsycho_assets("NURBS Patch")
        if not template_obj: return None
        collection=bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
        if col_name not in self.context.scene.collection.children: self.context.scene.collection.children.link(collection)
        for obj in collection.objects: bpy.data.objects.remove(obj,do_unlink=True)
        for face_idx, cps in patches.items():
            control_points = [self.matrix_world @ cp for row in reversed(cps) for cp in row]
            faces = [(v * 4 + u, v * 4 + u + 1, (v + 1) * 4 + u + 1, (v + 1) * 4 + u) for v in range(3) for u in range(3)]
            mesh = bpy.data.meshes.new(name=f"PsychoPatch_Mesh_{face_idx:04d}")
            mesh.from_pydata(control_points, [], faces);mesh.update()
            new_obj = template_obj.copy();new_obj.data = mesh;new_obj.name = f"{self.obj.name}_PsychoPatch_{face_idx:04d}"
            collection.objects.link(new_obj);set_first_vertex_smooth(new_obj)
        if template_obj.name in bpy.data.objects and template_obj.users == 1: bpy.data.objects.remove(template_obj)
        return collection
    def create_nurbs_template(self):
        bpy.ops.surface.primitive_nurbs_surface_surface_add(radius=1,location=(9e9,9e9,9e9))
        obj=bpy.context.active_object;data=obj.data.copy();spline=data.splines[0]
        spline.use_endpoint_u,spline.use_endpoint_v=True,True;spline.order_u,spline.order_v=4,4
        bpy.data.objects.remove(obj);return data
    def get_evaluated_bmesh(self):
        deps=bpy.context.evaluated_depsgraph_get();eval_obj=self.obj.evaluated_get(deps)
        mesh=bpy.data.meshes.new_from_object(eval_obj);bm=bmesh.new();bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table();bm.edges.ensure_lookup_table()
        bm.verts.index_update();bm.edges.index_update();bm.faces.index_update();return bm
    def walk_path(self, start_vert, end_vert, bm):
        if start_vert == end_vert: return [start_vert]
        distances = {v: float('inf') for v in bm.verts}; previous_verts = {v: None for v in bm.verts}
        distances[start_vert] = 0; counter = 0; pq = [(0, counter, start_vert)]
        while pq:
            current_distance, _, current_vert = heapq.heappop(pq)
            if current_vert == end_vert: break
            if current_distance > distances[current_vert]: continue
            for edge in current_vert.link_edges:
                neighbor = edge.other_vert(current_vert); distance = current_distance + edge.calc_length()
                if distance < distances[neighbor]:
                    distances[neighbor] = distance; previous_verts[neighbor] = current_vert; counter += 1
                    heapq.heappush(pq, (distance, counter, neighbor))
        path = []; current = end_vert
        while current is not None: path.append(current); current = previous_verts[current]
        if path and path[-1] == start_vert: return path[::-1]
        else: return None
    def _try_build_grid(self, path1_start, path1_end, path2_start, path2_end, bm_eval):
        path1 = self.walk_path(path1_start, path1_end, bm_eval); path2 = self.walk_path(path2_start, path2_end, bm_eval)
        if not path1 or not path2 or len(path1) != len(path2): return None
        m = len(path1)
        if m < 2: return None
        grid = []; first_row = self.walk_path(path1[0], path2[0], bm_eval)
        if not first_row: return None
        n = len(first_row)
        if n < 2: return None
        grid.append(first_row)
        for i in range(1, m):
            row = self.walk_path(path1[i], path2[i], bm_eval)
            if not row or len(row) != n: return None
            grid.append(row)
        return grid, m, n
    def trace_face_grid(self,f,vm,bm_e):
        try:
            cs=[l.vert for l in f.loops];v0,v1,v2,v3=[vm[v.index] for v in cs]
        except (KeyError,IndexError): return None
        result = self._try_build_grid(v0, v3, v1, v2, bm_e)
        if result: grid, m, n = result; return [[v.co for v in row] for row in grid], 0, m, n
        result = self._try_build_grid(v0, v1, v3, v2, bm_e)
        if result: grid, m, n = result; return [[v.co for v in row] for row in [[grid[j][i] for j in range(n)] for i in range(m)]], 1, n, m
        return None
    def fit_cps_to_grid(self,point_grid, m, n):
        if m==0 or n==0: return None
        num_cps_u, num_cps_v = 4, 4; Q_list=[co for r in point_grid for co in r]
        if len(Q_list)!=m*n:return None
        Q=np.array([[v.x,v.y,v.z] for v in Q_list])
        def bernstein(n_param,i,t):from math import comb;return comb(n_param,i)*(t**i)*((1-t)**(n_param-i))
        t_u = np.linspace(0,1,m); t_v = np.linspace(0,1,n); B_u = np.zeros((m, num_cps_u))
        for i in range(num_cps_u):
            for j in range(m): B_u[j,i]=bernstein(num_cps_u-1,i,t_u[j])
        B_v = np.zeros((n, num_cps_v))
        for i in range(num_cps_v):
            for j in range(n): B_v[j,i]=bernstein(num_cps_v-1,i,t_v[j])
        N = np.kron(B_v, B_u)
        try:P_flat=np.linalg.pinv(N)@Q
        except np.linalg.LinAlgError:return None
        return[[Vector(P_flat[r*num_cps_v+c]) for c in range(num_cps_v)] for r in range(num_cps_u)]

# --- UI y Registro ---
class SUBDIV_OT_convert_to_nurbs(bpy.types.Operator):
    bl_idname="subdiv.convert_to_nurbs";bl_label="Convert SubD";bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):
        obj=context.active_object
        return obj and obj.type=='MESH' and any(m.type=='SUBSURF' for m in obj.modifiers)
    def execute(self,context):
        obj=context.active_object;settings=context.scene.nurbs_converter_settings
        try:
            fitter=LimitSurfaceFitter(context,obj,settings);fitter.report=self.report
            if(fitter.run_conversion()):self.report({'INFO'},"Conversion complete.")
            else:self.report({'WARNING'},"NURBS conversion failed.")
        except Exception as e:
            self.report({'ERROR'},f"Unexpected error: {e}");import traceback;traceback.print_exc()
            return{'CANCELLED'}
        return{'FINISHED'}

class SUBDIV_PT_converter_panel(bpy.types.Panel):
    bl_label="SubD to NURBS / PsychoPatch";bl_idname="SUBDIV_PT_converter_panel";bl_space_type='VIEW_3D';bl_region_type='UI';bl_category="SubD to NURBS"
    def draw(self,context):
        layout = self.layout;obj = context.active_object
        if not obj or obj.type != 'MESH':
            layout.label(text="Select a Mesh object.", icon='ERROR');return
        try:
            mod = next(m for m in obj.modifiers if m.type == 'SUBSURF')
            layout.label(text=f"Objeto: {obj.name}", icon='MESH_DATA')
            box = layout.box();box.prop(mod, "levels", text="Subdivision Level")
            settings = context.scene.nurbs_converter_settings
            box = layout.box(); box.label(text="Output Type")
            box.prop(settings, "output_type", expand=True)
            if settings.output_type == 'BLENDER_NURBS':
                post_box = layout.box()
                post_box.label(text="Post-process modifiers:")
                post_box.label(text="Warning: Compromises accuracy.", icon='ERROR')
                col = post_box.column(align=True)
                col.prop(settings, "join_patches")
                sub_col = col.column(align=True); sub_col.active = settings.join_patches
                sub_col.prop(settings, "add_weld_modifier"); sub_col.prop(settings, "add_subd_modifier"); sub_col.prop(settings, "add_smooth_modifier")
            adv_box = layout.box();adv_box.label(text="Correction Parameters", icon='SETTINGS')
            col = adv_box.column(align=True)
            col.prop(settings, "generate_ruled_surfaces", text="Generate Ruled Surfaces")
            row = col.row(align=True);row.prop(settings, "force_planar_faces")
            sub_row = row.row(align=True);sub_row.active = settings.force_planar_faces;sub_row.prop(settings, "planar_tolerance", text="Threshold")
            col.prop(settings, "g1_transition_weight")
            col.prop(settings, "exceptional_weight");col.prop(settings, "fairing_weight")
            layout.separator();layout.operator("subdiv.convert_to_nurbs", icon='SURFACE_DATA')
        except StopIteration:
            layout.label(text="Add a SubD modifier.", icon='MOD_SUBSURF')

classes=(NurbsConverterSettings,SUBDIV_OT_convert_to_nurbs,SUBDIV_PT_converter_panel,)
def register():
    for cls in classes:bpy.utils.register_class(cls)
    bpy.types.Scene.nurbs_converter_settings=bpy.props.PointerProperty(type=NurbsConverterSettings)
def unregister():
    del bpy.types.Scene.nurbs_converter_settings
    for cls in reversed(classes):bpy.utils.unregister_class(cls)
if __name__=="__main__":
    register()