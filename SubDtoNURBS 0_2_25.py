#
# Full addon code with the new output material assignment feature.
# - Added an `output_material` PointerProperty to the settings.
# - The UI panel now displays this material selector in a shared "Post-process" section.
# - Both `_create_nurbs_objects` and `_create_psychopatch_objects` methods
#   have been updated to append the selected material to the newly created objects.
#

bl_info = {
    "name": "SubD to NURBS (v0.2.25-Output material support)",
    "author": "oobma, Gemini (Final Architect) DeepSeek, Qwen (Consultants)",
    "version": (0, 2, 25),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > SubDtoNURBS",
    "description": "Convert subdivision surfaces to NURBS with robust fallback handling",
    "category": "Object",
}

import bpy
import bmesh
from mathutils import Vector
import time
import numpy as np
import heapq
from pathlib import Path

# --- Helper Functions (unchanged) ---
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

# --- Settings ---
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
    psychopatch_extract_normals: bpy.props.BoolProperty(name="Extract Normals", description="Extract and store face corner normals for advanced shading (PsychoPatch only).", default=False)
    # NEW PROPERTY FOR THE OUTPUT MATERIAL
    output_material: bpy.props.PointerProperty(name="Output Material", type=bpy.types.Material, description="Apply this material to the created objects")

# --- Algorithm Classes (GeometryAnalyzer and EdgeCentricStitcher are unchanged) ---
class GeometryAnalyzer:
    def __init__(self, bm, exceptional_weight, planar_tolerance):
        self.bm=bm;self.w=exceptional_weight;self.ev={v.index for v in self.bm.verts if len(v.link_edges)!=4 and not v.is_boundary};self.planar_extraordinary_verts={};self._detect_planar_ev(planar_tolerance)
    def get_g1_constraint_weight(self,v_idx):return self.w if v_idx in self.ev else 1.0
    def _detect_planar_ev(self,tolerance):
        self.bm.verts.ensure_lookup_table()
        for v_idx in self.ev:
            v=self.bm.verts[v_idx];neighbors=[e.other_vert(v) for e in v.link_edges]
            if len(neighbors)<3:continue
            plane_co=v.co;plane_no=(neighbors[0].co-plane_co).cross(neighbors[1].co-plane_co)
            if plane_no.length_squared<1e-12:continue
            plane_no.normalize();is_planar=True
            for i in range(2,len(neighbors)):
                if abs((neighbors[i].co-plane_co).dot(plane_no))>tolerance:is_planar=False;break
            if is_planar:self.planar_extraordinary_verts[v_idx]=(plane_co,plane_no)

class EdgeCentricStitcher:
    def __init__(self, bm, cache, obj, settings, planar_map, ruled_map, crease_layer):
        self.bm, self.cache, self.obj, self.settings = bm, cache, obj, settings
        self.planar_map=planar_map; self.ruled_map=ruled_map; self.crease_layer=crease_layer
        self.initial={i:[r[:] for r in v['cps']] for i,v in self.cache.items()}
        self.final={i:[r[:] for r in v['cps']] for i,v in self.cache.items()}
        self.map={e.index:[] for e in self.bm.edges}
        self.geo=GeometryAnalyzer(self.bm,self.settings.exceptional_weight,self.settings.planar_tolerance)
    def run(self):
        self._build_map();deps=bpy.context.evaluated_depsgraph_get();obj_eval=self.obj.evaluated_get(deps)
        self.verts=[v.co.copy() for v in obj_eval.data.vertices];self._enforce_G0();self._solve_G1()
        return self.final
    def _get_side_map(self, f, o):
        v0,v1,v2,v3=(l.vert for l in f.loops);et,er,eb,el=find_edge_between_verts(v0,v1),find_edge_between_verts(v1,v2),find_edge_between_verts(v2,v3),find_edge_between_verts(v3,v0)
        return{'U0':et,'U1':eb,'V0':el,'V1':er} if o==0 else{'U0':el,'U1':er,'V0':et,'V1':eb}
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
                fa_is_special=self.planar_map.get(fa_idx,False) or self.ruled_map.get(fa_idx,False);fb_is_special=self.planar_map.get(fb_idx,False) or self.ruled_map.get(fb_idx,False)
                if fa_is_special and not fb_is_special:g0_boundary=self._get_b(self.initial[fa_idx],sa)
                elif fb_is_special and not fa_is_special:g0_boundary=self._get_b(self.initial[fb_idx],sb,reverse=True)
            if g0_boundary is None:
                pa,pb=self.initial[fa_idx],self.initial[fb_idx];ca,cb=self._get_b(pa,sa),self._get_b(pb,sb,reverse=True)
                g0_boundary=[(p1+p2)/2.0 for p1,p2 in zip(ca,cb)]
            vs,ve=e.verts;cos,coe=self.verts[vs.index],self.verts[ve.index]
            if(g0_boundary[0]-cos).length_squared>(g0_boundary[0]-coe).length_squared:cos,coe=coe,cos
            g0_boundary[0],g0_boundary[3]=cos,coe
            self._set_b(self.final[fa_idx],sa,g0_boundary);self._set_b(self.final[fb_idx],sb,g0_boundary,reverse=True)
    def _solve_G1(self):
        v_map={(f,r,c):i for i,(f,r,c) in enumerate((fi,r,c) for fi in self.initial for r in range(1,3) for c in range(1,3))}
        if not v_map:return
        rows,b_vectors=[],[]
        for e in self.bm.edges:
            if e.is_boundary or len(self.map.get(e.index,[]))!=2:continue
            (fa,sa),(fb,sb)=self.map[e.index];ca,cb,bd=self._get_i(sa),self._get_i(sb,reverse=True),self._get_bi(sa)
            vs,ve=e.verts;is_transition=self.planar_map.get(fa,False)!=self.planar_map.get(fb,False)
            for i in range(4):
                w=1.0
                if is_transition:w*=self.settings.g1_transition_weight
                if i==0:w*=self.geo.get_g1_constraint_weight(vs.index)
                elif i==3:w*=self.geo.get_g1_constraint_weight(ve.index)
                rb,cb_idx=bd[i];b_val=w*(2.0*self.final[fa][rb][cb_idx]);row=[]
                ar,ac=ca[i]
                if(fa,ar,ac) in v_map:row.append((v_map[(fa,ar,ac)],w))
                else:b_val-=w*self.final[fa][ar][ac]
                br,bc=cb[i]
                if(fb,br,bc) in v_map:row.append((v_map[(fb,br,bc)],w))
                else:b_val-=w*self.final[fb][br][bc]
                if row:rows.append(row);b_vectors.append(b_val)
        wf=self.settings.fairing_weight
        if wf>0:
            for(f,r,c),i in v_map.items():rows.append([(i,wf)]);b_vectors.append(wf*self.initial[f][r][c])
        w_planar=10.0
        for v_idx,(plane_co,plane_no) in self.geo.planar_extraordinary_verts.items():
            v=self.bm.verts[v_idx]
            for f in v.link_faces:
                if f.index not in self.cache:continue
                try:
                    face_verts=[l.vert.index for l in f.loops];corner_idx=face_verts.index(v_idx);o=self.cache[f.index]['orientation']
                    rc_map_o0=[(1,1),(1,2),(2,2),(2,1)];rc_map_o1=[(1,1),(2,1),(2,2),(1,2)];r,c=rc_map_o0[corner_idx] if o==0 else rc_map_o1[corner_idx]
                    cp_tuple=(f.index,r,c)
                    if cp_tuple in v_map:
                        i=v_map[cp_tuple];initial_pos=self.initial[f.index][r][c];dist=(initial_pos-plane_co).dot(plane_no)
                        projected_pos=initial_pos-dist*plane_no;rows.append([(i,w_planar)]);b_vectors.append(w_planar*projected_pos)
                except ValueError:continue
        n=len(v_map);A=np.zeros((n,n));b_np=np.array([[vec.x,vec.y,vec.z] for vec in b_vectors]);Atb=np.zeros((n,3))
        for i,row in enumerate(rows):
            for i1,v1 in row:
                Atb[i1]+=v1*b_np[i]
                for i2,v2 in row:A[i1,i2]+=v1*v2
        sol_x,sol_y,sol_z=conjugate_gradient_solver(A,Atb[:,0]),conjugate_gradient_solver(A,Atb[:,1]),conjugate_gradient_solver(A,Atb[:,2])
        for(f,r,c),i in v_map.items():self.final[f][r][c]=Vector((sol_x[i],sol_y[i],sol_z[i]))
    
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
    def _set_i(self,p,s,c,reverse=False):
        cp=c[::-1] if reverse else c[:];
        if s=='U0':
            for i in range(4):p[1][i]=cp[i]
        elif s=='U1':
            for i in range(4):p[2][3-i]=cp[i]
        elif s=='V0':
            for i in range(4):p[3-i][1]=cp[i]
        elif s=='V1':
            for i in range(4):p[i][2]=cp[i]

class LimitSurfaceFitter:
    def __init__(self,context,obj,settings):
        self.context,self.obj,self.matrix_world=context,obj,obj.matrix_world.copy();self.settings=settings;self.template_data=self.create_nurbs_template();self.report=lambda*args,**kwargs:None
        self.oriented_fit_cache={};self.ruled_map={};self.planar_map={};self.fallback_face_indices=set()
        
    def run_conversion(self):
        start_time=time.time();print(f"Starting NURBS conversion ({bl_info['name']})...")
        bm_orig=bmesh.new();bm_orig.from_mesh(self.obj.data)
        deps=bpy.context.evaluated_depsgraph_get();self.obj_eval=self.obj.evaluated_get(deps)
        bm_eval=self.get_evaluated_bmesh()
        crease_layer=bm_orig.edges.layers.float.get("crease_edge") or bm_orig.edges.layers.float.new("crease_edge")
        bm_orig.verts.ensure_lookup_table();bm_orig.faces.ensure_lookup_table();bm_orig.edges.ensure_lookup_table()
        orig_vert_to_eval_map={ov.index:ev for ov,ev in zip(bm_orig.verts,bm_eval.verts)}
        print("Phase 1: Global initial setting...")
        self.oriented_fit_cache={};self.ruled_map={};self.fallback_face_indices=set()
        
        for f in bm_orig.faces:
            if len(f.verts)!=4:continue
            trace_result=self.trace_face_grid(f,orig_vert_to_eval_map,bm_eval)
            if trace_result is None:
                print(f"   - Path tracer failed for face {f.index}, using projection fallback.")
                trace_result=self.trace_face_grid_projected(f,orig_vert_to_eval_map);self.fallback_face_indices.add(f.index)
            if not trace_result:
                print(f"   - All tracers failed for face {f.index}. Skipping.");continue
            point_grid,orientation,m,n=trace_result
            control_points=self.fit_cps_to_grid(point_grid,m,n)
            if not control_points:continue
            
            self.oriented_fit_cache[f.index]={'cps':control_points,'orientation':orientation}
            if self.settings.generate_ruled_surfaces and self.check_if_ruled_face(f,crease_layer):
                self.ruled_map[f.index]=True

        if not self.oriented_fit_cache:
            self.report({'ERROR'},"Initial adjustment failure.");bm_orig.free();bm_eval.free();return None
        
        print("   - Checking all patches for flatness...")
        self.planar_map={idx:is_patch_truly_flat(data['cps'],self.settings.planar_tolerance) for idx,data in self.oriented_fit_cache.items()}
        print("Phase 2: Global sewing...")
        stitcher=EdgeCentricStitcher(bm_orig,self.oriented_fit_cache,self.obj,self.settings,self.planar_map,self.ruled_map,crease_layer)
        final_patches=stitcher.run()
        
        if self.fallback_face_indices:
            print("Phase 2.1: Correcting fallback patch boundaries and tangents...")
            final_patches=self.correct_fallback_patch_boundaries(final_patches,bm_orig,stitcher)
        if self.settings.generate_ruled_surfaces and self.ruled_map:
            print("Phase 2.2: Enforcing ruled surface linearity...")
            final_patches=self.enforce_ruled_surfaces(final_patches,bm_orig,crease_layer)
        if self.settings.force_planar_faces:
            print("Phase 2.5: Correction for design intent...")
            final_patches=self.get_planarized_patches(final_patches)
        print(f"Phase 3: Creating objects type '{self.settings.output_type}'...")
        collection=None
        if self.settings.output_type=='BLENDER_NURBS':collection=self._create_nurbs_objects(final_patches,f"{self.obj.name}_NURBS_Patches")
        elif self.settings.output_type=='PSYCHOPATCH':
            collection=self._create_psychopatch_objects(final_patches,f"{self.obj.name}_Psycho_Patches")
            if not collection:self.report({'ERROR'},"Unable to load ‘NURBS Patch’ from surfacepsycho.")
        bm_orig.free();bm_eval.free();print(f"✅ NURBS conversion completed in {time.time()-start_time:.2f}s.")
        return collection

    def correct_fallback_patch_boundaries(self,final_patches,bm,stitcher):
        corrected_patches={idx:[[cp.copy() for cp in row] for row in cps] for idx,cps in final_patches.items()}
        corrected_count=0
        edge_map={e.index:[f.index for f in e.link_faces] for e in bm.edges}
        for fallback_idx in self.fallback_face_indices:
            if fallback_idx not in corrected_patches:continue
            face=bm.faces[fallback_idx]
            fallback_cps=corrected_patches[fallback_idx]
            for edge in face.edges:
                neighbors=edge_map.get(edge.index,[])
                if len(neighbors)!=2:continue
                neighbor_idx=neighbors[0] if neighbors[1]==fallback_idx else neighbors[1]
                if neighbor_idx in self.fallback_face_indices or neighbor_idx not in corrected_patches:continue
                side_fallback,side_neighbor=None,None
                for f_idx,s in stitcher.map[edge.index]:
                    if f_idx==fallback_idx:side_fallback=s
                    elif f_idx==neighbor_idx:side_neighbor=s
                if side_fallback and side_neighbor:
                    reliable_patch=corrected_patches[neighbor_idx]
                    reliable_boundary=stitcher._get_b(reliable_patch,side_neighbor,reverse=True)
                    stitcher._set_b(fallback_cps,side_fallback,reliable_boundary,reverse=False)
                    reliable_interior_indices=stitcher._get_i(side_neighbor,reverse=True)
                    reliable_interior_row=[reliable_patch[r][c] for r,c in reliable_interior_indices]
                    new_interior_row=[(2*b_cp)-i_cp for b_cp,i_cp in zip(reliable_boundary,reliable_interior_row)]
                    stitcher._set_i(fallback_cps,side_fallback,new_interior_row,reverse=False)
            P=fallback_cps
            for i in range(1,3):
                for j in range(1,3):
                    u,v=i/3.0,j/3.0
                    pu=(1-u)*P[0][j]+u*P[3][j]
                    pv=(1-v)*P[i][0]+v*P[i][3]
                    pc=(1-u)*(1-v)*P[0][0]+u*(1-v)*P[3][0]+(1-u)*v*P[0][3]+u*v*P[3][3]
                    fallback_cps[i][j]=pu+pv-pc
            corrected_count+=1
        if corrected_count>0:print(f"     - Corrected G0/G1 boundaries on {corrected_count} fallback patches.")
        return corrected_patches
    def _calculate_grid_length(self, grid):
        length = 0.0; n, m = len(grid), len(grid[0])
        for i in range(n):
            for j in range(m - 1): length += (grid[i][j] - grid[i][j+1]).length
        for j in range(m):
            for i in range(n - 1): length += (grid[i][j] - grid[i+1][j]).length
        return length
    def trace_face_grid_projected(self, f, orig_vert_to_eval_map):
        try:
            loops = list(f.loops); v0e, v1e, v2e, v3e = (orig_vert_to_eval_map[l.vert.index].co.copy() for l in loops)
        except KeyError: return None
        m, n = 8, 8
        grid_u_guess = [[v0e.lerp(v1e, u).lerp(v3e.lerp(v2e, u), v) for u in np.linspace(0,1,m)] for v in np.linspace(0,1,n)]
        len_u = self._calculate_grid_length(grid_u_guess)
        grid_v_guess_raw = [[v0e.lerp(v3e, v).lerp(v1e.lerp(v2e, v), u) for u in np.linspace(0,1,m)] for v in np.linspace(0,1,n)]
        len_v = self._calculate_grid_length(grid_v_guess_raw)
        best_guess_grid, orientation = (grid_u_guess, 0) if len_u <= len_v else (grid_v_guess_raw, 1)
        final_grid = []
        rows, cols = (n, m)
        for row_idx in range(rows):
            new_row = []
            for col_idx in range(cols):
                p_guess_local = best_guess_grid[row_idx][col_idx]
                p_guess_world = self.matrix_world @ p_guess_local
                result, location, norm, face_idx = self.obj_eval.closest_point_on_mesh(p_guess_world)
                if result: new_row.append(self.matrix_world.inverted() @ location)
                else: new_row.append(p_guess_local)
            if len(new_row) == cols: final_grid.append(new_row)
        if len(final_grid) != rows: return None
        if orientation == 1:
            final_grid = [[final_grid[j][i] for j in range(rows)] for i in range(cols)]; m, n = n, m
        return final_grid, orientation, m, n
    def enforce_ruled_surfaces(self, final_patches, bm, crease_layer):
        print("   - Post-processing ruled surfaces for linearity..."); corrected_patches = {idx: [[cp.copy() for cp in row] for row in cps] for idx, cps in final_patches.items()}; ruled_count = 0
        for face_idx in self.ruled_map:
            if face_idx not in corrected_patches: continue
            patch_cps = corrected_patches[face_idx]; face = bm.faces[face_idx]; o = self.oriented_fit_cache[face_idx]['orientation']
            loops = list(face.loops); edges = [l.edge for l in loops]; creases = [e[crease_layer] for e in edges]
            is_v_ruled = (o == 0 and creases[0] > 0.99 and creases[2] > 0.99) or (o == 1 and creases[1] > 0.99 and creases[3] > 0.99)
            is_u_ruled = (o == 0 and creases[1] > 0.99 and creases[3] > 0.99) or (o == 1 and creases[0] > 0.99 and creases[2] > 0.99)
            if is_v_ruled:
                ruled_count += 1; P0 = patch_cps[0]; P3 = patch_cps[3]
                for j in range(4): patch_cps[1][j] = (P0[j]*2+P3[j])/3.0; patch_cps[2][j] = (P0[j]+P3[j]*2)/3.0
            elif is_u_ruled:
                ruled_count += 1
                for i in range(4):
                    P_i0=patch_cps[i][0]; P_i3=patch_cps[i][3]; patch_cps[i][1]=(P_i0*2+P_i3)/3.0; patch_cps[i][2]=(P_i0+P_i3*2)/3.0
        print(f"     - Linearity enforced on {ruled_count} ruled patches."); return corrected_patches
    def check_if_ruled_face(self, face, crease_layer):
        loops=list(face.loops);edges=[l.edge for l in loops];creases=[e[crease_layer] for e in edges]
        if creases[0]>0.99 and creases[2]>0.99 and creases[1]<0.01 and creases[3]<0.01:return True
        if creases[1]>0.99 and creases[3]>0.99 and creases[0]<0.01 and creases[2]<0.01:return True
        return False
    def get_planarized_patches(self, stitched_patches):
        print("   - Applying post-processing to force planarity...");new_patches={};planar_faces_count=0
        for face_idx,final_cps in stitched_patches.items():
            new_control_points=[[cp.copy() for cp in row] for row in final_cps]
            if self.planar_map.get(face_idx, False):
                planar_faces_count+=1;p00, p03, p30 = final_cps[0][0], final_cps[0][3], final_cps[3][0]
                plane_co=p00;plane_no=(p30 - p00).cross(p03-p00)
                if plane_no.length_squared > 1e-12:
                    plane_no.normalize()
                    for r in range(4):
                        for c in range(4):dist=(new_control_points[r][c]-plane_co).dot(plane_no);new_control_points[r][c]-=dist*plane_no
            new_patches[face_idx] = new_control_points
        print(f"     - Flatness forcing applied to {planar_faces_count} patches.");return new_patches
    def create_nurbs_template(self):
        bpy.ops.surface.primitive_nurbs_surface_surface_add(radius=1, location=(9e9, 9e9, 9e9))
        obj = bpy.context.active_object
        data = obj.data.copy()
        spline = data.splines[0]
        spline.use_endpoint_u, spline.use_endpoint_v = True, True
        spline.order_u, spline.order_v = 4, 4
        bpy.data.objects.remove(obj)
        return data
    def _create_nurbs_objects(self,patches, col_name):
        nurbs_col=bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
        if col_name not in self.context.scene.collection.children:self.context.scene.collection.children.link(nurbs_col)
        for obj in nurbs_col.objects:bpy.data.objects.remove(obj,do_unlink=True)
        created_objects = []
        for face_idx,cps in patches.items():
            name,nurbs_data=f"{self.obj.name}_Patch_{face_idx:04d}",self.template_data.copy();spline=nurbs_data.splines[0]
            flat_cps=[cp for row in reversed(cps) for cp in row]
            for i,pt in enumerate(spline.points):pt.co=(*flat_cps[i],1.0)
            nurbs_obj=bpy.data.objects.new(name,nurbs_data)
            nurbs_obj.matrix_world = self.matrix_world
            nurbs_col.objects.link(nurbs_obj)
            if self.settings.output_material:
                nurbs_obj.data.materials.append(self.settings.output_material)
            created_objects.append(nurbs_obj)
        if self.settings.join_patches and created_objects:
            print("   - Joining patches and adding modifiers...");base_obj = created_objects[0]
            override_context = self.context.copy();override_context['active_object'] = base_obj;override_context['selected_editable_objects'] = created_objects
            with bpy.context.temp_override(**override_context): bpy.ops.object.join()
            joined_obj = base_obj
            if joined_obj and joined_obj.name in bpy.data.objects:
                joined_obj.name = f"{self.obj.name}_Joined_NURBS"
                if self.settings.add_weld_modifier: joined_obj.modifiers.new(name="Weld", type='WELD').merge_threshold=0.0001
                if self.settings.add_subd_modifier: joined_obj.modifiers.new(name="Subdivision", type='SUBSURF').levels=1
                if self.settings.add_smooth_modifier: joined_obj.modifiers.new(name="Smooth", type='SMOOTH').iterations=1
        return nurbs_col
    def _create_psychopatch_objects(self, patches, col_name):
        template_obj = bpy.data.objects.get("NURBS Patch") or append_object_from_surfacepsycho_assets("NURBS Patch")
        if not template_obj: return None
        collection=bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
        if col_name not in self.context.scene.collection.children: self.context.scene.collection.children.link(collection)
        for obj in collection.objects: bpy.data.objects.remove(obj,do_unlink=True)
        for face_idx, cps in patches.items():
            control_points = [cp for row in reversed(cps) for cp in row]
            faces = [(v*4+u, v*4+u+1, (v+1)*4+u+1, (v+1)*4+u) for v in range(3) for u in range(3)]
            mesh = bpy.data.meshes.new(name=f"PsychoPatch_Mesh_{face_idx:04d}");mesh.from_pydata(control_points, [], faces);mesh.update()
            new_obj = template_obj.copy();new_obj.data = mesh;new_obj.name = f"{self.obj.name}_PsychoPatch_{face_idx:04d}"
            new_obj.matrix_world = self.matrix_world
            collection.objects.link(new_obj);set_first_vertex_smooth(new_obj)
            if self.settings.output_material:
                new_obj.data.materials.append(self.settings.output_material)
            if self.settings.psychopatch_extract_normals:
                mod = new_obj.modifiers.get("SP - NURBS Patch Meshing")
                if mod:
                    try: mod["Socket_21"] = True
                    except (KeyError, TypeError): print(f"Warning: Could not set 'Extract Normals' on {new_obj.name}")
        if template_obj.name in bpy.data.objects and template_obj.users == 1: bpy.data.objects.remove(template_obj)
        return collection
    def get_evaluated_bmesh(self):
        mesh=bpy.data.meshes.new_from_object(self.obj_eval);bm=bmesh.new();bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table();bm.faces.ensure_lookup_table();bm.edges.ensure_lookup_table()
        bm.verts.index_update();bm.edges.index_update();bm.faces.index_update();return bm
    def walk_path(self, start_vert, end_vert, bm):
        if start_vert == end_vert: return [start_vert]
        distances = {v: float('inf') for v in bm.verts}; previous_verts = {v: None for v in bm.verts}
        distances[start_vert] = 0; counter = 0; pq = [(0, counter, start_vert)]
        while pq:
            dist, _, curr = heapq.heappop(pq)
            if curr == end_vert: break
            if dist > distances[curr]: continue
            for edge in curr.link_edges:
                neighbor = edge.other_vert(curr); d = dist + edge.calc_length()
                if d < distances[neighbor]:
                    distances[neighbor] = d; previous_verts[neighbor] = curr; counter += 1
                    heapq.heappush(pq, (d, counter, neighbor))
        path = []; current = end_vert
        while current is not None: path.append(current); current = previous_verts[current]
        if path and path[-1] == start_vert: return path[::-1]
        else: return None
    def _try_build_grid(self, p1_start, p1_end, p2_start, p2_end, bm_eval):
        path1 = self.walk_path(p1_start, p1_end, bm_eval); path2 = self.walk_path(p2_start, p2_end, bm_eval)
        if not path1 or not path2 or len(path1) != len(path2): return None
        m = len(path1);
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
        try:cs=[l.vert for l in f.loops];v0,v1,v2,v3=[vm[v.index] for v in cs]
        except(KeyError,IndexError): return None
        result = self._try_build_grid(v0,v3,v1,v2,bm_e)
        if result: grid,m,n=result; return [[v.co for v in row] for row in grid],0,m,n
        result = self._try_build_grid(v0,v1,v3,v2,bm_e)
        if result: grid,m,n=result; return [[v.co for v in row] for row in [[grid[j][i] for j in range(m)] for i in range(n)]],1,n,m
        return None
    def fit_cps_to_grid(self,point_grid, m, n):
        if m==0 or n==0: return None
        num_cps_u,num_cps_v=4,4; Q_list=[co for r in point_grid for co in r]
        if len(Q_list)!=m*n:return None
        Q=np.array([[v.x,v.y,v.z] for v in Q_list])
        def bernstein(n_param,i,t):from math import comb;return comb(n_param,i)*(t**i)*((1-t)**(n_param-i))
        t_u=np.linspace(0,1,m);t_v=np.linspace(0,1,n);B_u=np.zeros((m,num_cps_u));B_v=np.zeros((n,num_cps_v))
        for i in range(num_cps_u):
            for j in range(m): B_u[j,i]=bernstein(num_cps_u-1,i,t_u[j])
        for i in range(num_cps_v):
            for j in range(n): B_v[j,i]=bernstein(num_cps_v-1,i,t_v[j])
        N=np.kron(B_v,B_u)
        try:P_flat=np.linalg.pinv(N)@Q
        except np.linalg.LinAlgError:return None
        return[[Vector(P_flat[r*num_cps_v+c]) for c in range(num_cps_v)] for r in range(num_cps_u)]

# --- UI and Registration ---
class SUBDIV_OT_convert_to_nurbs(bpy.types.Operator):
    bl_idname="subdiv.convert_to_nurbs";bl_label="Convert SubD to Patches";bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):obj=context.active_object;return obj and obj.type=='MESH' and any(m.type=='SUBSURF' for m in obj.modifiers)
    def execute(self,context):
        obj=context.active_object;settings=context.scene.nurbs_converter_settings
        if settings.output_type == 'PSYCHOPATCH' and settings.psychopatch_extract_normals:
            if not context.scene.render.use_high_quality_normals:
                context.scene.render.use_high_quality_normals = True
                self.report({'INFO'}, "Enabled 'High Quality Normals' for PsychoPatch.")
                print("Info: Enabled 'High Quality Normals' for PsychoPatch normal extraction.")
        try:
            fitter=LimitSurfaceFitter(context,obj,settings);fitter.report=self.report
            if(fitter.run_conversion()):self.report({'INFO'},"Conversion complete.")
            else:self.report({'WARNING'},"NURBS conversion failed. Check console for details.")
        except Exception as e:
            self.report({'ERROR'},f"Unexpected error: {e}");import traceback;traceback.print_exc()
            return{'CANCELLED'}
        return{'FINISHED'}

class SUBDIV_PT_converter_panel(bpy.types.Panel):
    bl_label="SubD to NURBS / PsychoPatch";bl_idname="SUBDIV_PT_converter_panel";bl_space_type='VIEW_3D';bl_region_type='UI';bl_category="SubD to NURBS"
    def draw(self,context):
        layout=self.layout;obj=context.active_object
        if not obj or obj.type!='MESH':layout.label(text="Select a Mesh object.",icon='ERROR');return
        try:
            mod=next(m for m in obj.modifiers if m.type=='SUBSURF')
            layout.label(text=f"Object: {obj.name}",icon='MESH_DATA');box=layout.box();box.prop(mod,"levels",text="Subdivision Level")
            settings=context.scene.nurbs_converter_settings;box=layout.box();box.label(text="Output Type");box.prop(settings,"output_type",expand=True)
            
            post_box = layout.box(); post_box.label(text="Post-process:")
            col = post_box.column(align=True)
            col.prop(settings, "output_material")

            if settings.output_type=='BLENDER_NURBS':
                col.prop(settings, "join_patches")
                sub_col = col.column(align=True); sub_col.active = settings.join_patches
                sub_col.prop(settings, "add_weld_modifier"); sub_col.prop(settings, "add_subd_modifier"); sub_col.prop(settings, "add_smooth_modifier")
            elif settings.output_type == 'PSYCHOPATCH':
                col.prop(settings, "psychopatch_extract_normals")

            adv_box=layout.box();adv_box.label(text="Correction Parameters",icon='SETTINGS')
            col=adv_box.column(align=True);col.prop(settings,"generate_ruled_surfaces",text="Generate Ruled Surfaces")
            row=col.row(align=True);row.prop(settings,"force_planar_faces")
            sub_row=row.row(align=True);sub_row.active=settings.force_planar_faces;sub_row.prop(settings,"planar_tolerance",text="Threshold")
            col.prop(settings,"g1_transition_weight");col.prop(settings,"exceptional_weight");col.prop(settings,"fairing_weight")
            layout.separator();layout.operator("subdiv.convert_to_nurbs",icon='SURFACE_DATA')
        except StopIteration:layout.label(text="Add a SubD modifier.",icon='MOD_SUBSURF')

classes=(NurbsConverterSettings,SUBDIV_OT_convert_to_nurbs,SUBDIV_PT_converter_panel,)
def register():
    for cls in classes:bpy.utils.register_class(cls)
    bpy.types.Scene.nurbs_converter_settings=bpy.props.PointerProperty(type=NurbsConverterSettings)
def unregister():
    del bpy.types.Scene.nurbs_converter_settings
    for cls in reversed(classes):bpy.utils.unregister_class(cls)
if __name__=="__main__":
    register()