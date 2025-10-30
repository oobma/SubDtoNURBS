bl_info = {
    "name": "SubD to NURBS (v0.1.30) (Definitive API)",
    "author": "oobma, Gemini (Final Architect & Optimizer)",
    "version": (0, 1, 30),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar > SubDtoNURBS | Edit > Preferences > Add-ons",
    "description": "Convert subdivision surfaces to NURBS and export to IGES.",
    "category": "Object",
    "doc_url": "https://github.com/oobma/SubDtoNURBS"
}

import bpy
import bmesh
from mathutils import Vector
import time
import numpy as np
import heapq
from pathlib import Path
from bpy.app.handlers import persistent
import datetime
import concurrent.futures
import sys
import os
import subprocess
import importlib

# --- Dependency Management (Unchanged) ---
SCIPY_AVAILABLE = False

def get_dependency_path():
    path = Path(bpy.utils.script_path_user()) / "modules"
    os.makedirs(path, exist_ok=True)
    return str(path)

def ensure_dependency_path():
    dep_path = get_dependency_path()
    if dep_path not in sys.path:
        sys.path.insert(0, dep_path)

def check_scipy():
    global SCIPY_AVAILABLE
    try:
        ensure_dependency_path()
        importlib.import_module("scipy.sparse")
        importlib.import_module("scipy.sparse.linalg")
        SCIPY_AVAILABLE = True
        print("SubDtoNURBS: SciPy backend is available and verified.")
    except ImportError:
        SCIPY_AVAILABLE = False
        print("SubDtoNURBS: SciPy backend not found or incomplete. Using standard NumPy backend.")

check_scipy()

# --- Helper Functions (Unchanged) ---
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
        extensions_path = Path(bpy.utils.resource_path('USER')) / "extensions" / "blender_org" / "surfacepsycho" / "assets" / "assets.blend"
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

# --- State Management for Auto-Update (Unchanged) ---
def set_auto_update_source_object(scene, source_obj):
    scene["_nurbs_source_obj_name"] = source_obj.name if source_obj else ""
    if source_obj:
        scene["_nurbs_source_obj_last_mode"] = source_obj.mode
def get_auto_update_source_object(context):
    scene = context.scene if hasattr(context, "scene") else context
    obj_name = scene.get("_nurbs_source_obj_name", "")
    if obj_name in bpy.data.objects:
        return bpy.data.objects[obj_name]
    return None
def clear_auto_update_state(context):
    scene = context.scene if hasattr(context, "scene") else context
    if scene.nurbs_converter_settings:
        scene.nurbs_converter_settings.is_auto_update_active = False
    if "_nurbs_source_obj_name" in scene: del scene["_nurbs_source_obj_name"]
    if "_nurbs_update_pending" in scene: del scene["_nurbs_update_pending"]
    if "_nurbs_source_obj_last_mode" in scene: del scene["_nurbs_source_obj_last_mode"]

# --- Handlers & Timer Logic (Unchanged) ---
is_running_update = False
def run_pending_nurbs_update():
    global is_running_update
    if is_running_update or not bpy.context.scene.get("_nurbs_update_pending", False): return None
    is_running_update = True
    bpy.context.scene["_nurbs_update_pending"] = False
    source_obj = get_auto_update_source_object(bpy.context)
    if source_obj and source_obj.name in bpy.context.view_layer.objects:
        if bpy.context.view_layer.objects.active != source_obj:
            bpy.context.view_layer.objects.active = source_obj
        try: bpy.ops.subdiv.convert_to_nurbs('EXEC_DEFAULT')
        except Exception as e: print(f"Error during automatic NURBS update: {e}")
    else:
        clear_auto_update_state(bpy.context.scene)
        ensure_handlers_are_registered(False)
    is_running_update = False
    return None
@persistent
def depsgraph_update_handler(scene, depsgraph):
    settings = scene.nurbs_converter_settings
    if not settings.is_auto_update_active or is_running_update: return
    source_obj = get_auto_update_source_object(scene)
    if not source_obj:
        clear_auto_update_state(scene)
        ensure_handlers_are_registered(False)
        return
    last_mode = scene.get("_nurbs_source_obj_last_mode", source_obj.mode)
    current_mode = source_obj.mode
    if last_mode == 'EDIT' and current_mode == 'OBJECT':
        if scene.get("_nurbs_update_pending", False):
            scene["_nurbs_source_obj_last_mode"] = current_mode
            return
        scene["_nurbs_update_pending"] = True
        if not bpy.app.timers.is_registered(run_pending_nurbs_update):
            bpy.app.timers.register(run_pending_nurbs_update, first_interval=0.01)
    scene["_nurbs_source_obj_last_mode"] = current_mode
def ensure_handlers_are_registered(register):
    is_registered = depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post
    if register and not is_registered:
        bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)
    elif not register and is_registered:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)

# --- Settings (Unchanged) ---
class NurbsConverterSettings(bpy.types.PropertyGroup):
    is_auto_update_active: bpy.props.BoolProperty(name="Is Auto-Update Active", default=False)
    output_type: bpy.props.EnumProperty(name="Output Type",description="Choose output type.",items=[('BLENDER_NURBS', "Blender NURBS", "" ),('PSYCHOPATCH', "PsychoPatch", "")],default='BLENDER_NURBS')
    join_patches: bpy.props.BoolProperty(name="Join Patches", description="Join all patches into a single object.", default=False)
    add_weld_modifier: bpy.props.BoolProperty(name="Add Weld", description="Add a Weld modifier to close micro-seams.", default=False)
    add_subd_modifier: bpy.props.BoolProperty(name="Add Subdivision", description="Add a Subdivision Surface modifier to refine the mesh.", default=False)
    add_smooth_modifier: bpy.props.BoolProperty(name="Add Smooth", description="Add a Smooth modifier for perfect shading.", default=False)
    generate_ruled_surfaces:bpy.props.BoolProperty(name="Correct Ruled Surfaces",description="Smooth transition between opposite folded edges with G1 continuity.",default=True)
    force_planar_faces:bpy.props.BoolProperty(name="Force Planar Faces",description="Post-process faces with creased boundaries to enforce perfect flatness.",default=True)
    planar_tolerance:bpy.props.FloatProperty(name="Planar Tolerance",description="Sensitivity for detecting a face as planar.",default=1e-4,min=1e-7,max=1e-2,subtype='FACTOR',precision=6)
    g1_transition_weight:bpy.props.FloatProperty(name="G1 Transition Weight",description="Weaken G1 between flat/curved patches to prevent bulging.",default=0.05,min=0.0,max=1.0,subtype='FACTOR')
    exceptional_weight:bpy.props.FloatProperty(name="Exceptional Weight",description="Control G1 strength at exceptional vertices to reduce denting.",default=0.1,min=0.0,max=1.0,subtype='FACTOR')
    fairing_weight:bpy.props.FloatProperty(name="Fairing Weight",description="Pull CPs towards their 'ideal' shape to stabilize the solution.",default=0.5,min=0.0,max=1.0,subtype='FACTOR')
    psychopatch_extract_normals: bpy.props.BoolProperty(name="Extract Normals", description="Extract and store face corner normals for advanced shading (PsychoPatch only).", default=False)
    output_material: bpy.props.PointerProperty(name="Output Material", type=bpy.types.Material, description="Apply this material to the created objects")

# --- Algorithm Classes (Heavily Optimized) ---
class GeometryAnalyzer:
    def __init__(self, bm, exceptional_weight, planar_tolerance):
        self.bm=bm;self.w=exceptional_weight;self.ev={v.index for v in self.bm.verts if len(v.link_edges)!=4 and not v.is_boundary};self.planar_extraordinary_verts={};self._detect_planar_ev(planar_tolerance)
    def get_g1_constraint_weight(self,v_idx):
        if v_idx in self.planar_extraordinary_verts: return 1.0
        return self.w if v_idx in self.ev else 1.0
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
    def __init__(self, bm, cache, obj, settings, planar_map, ruled_map, crease_data_np):
        self.bm, self.cache, self.obj, self.settings = bm, cache, obj, settings
        self.planar_map=planar_map; self.ruled_map=ruled_map; self.crease_data=crease_data_np
        self.initial={i:[r[:] for r in v['cps']] for i,v in self.cache.items()}
        self.final={i:[r[:] for r in v['cps']] for i,v in self.cache.items()}
        self.map={e.index:[] for e in self.bm.edges}
        self.geo=GeometryAnalyzer(self.bm,self.settings.exceptional_weight,self.settings.planar_tolerance)

    def run(self, eval_coords_np):
        self._build_map()
        self.verts = eval_coords_np
        self._enforce_G0()
        self._solve_G1()
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
            if e.is_boundary or len(self.map.get(e.index, [])) != 2: continue
            (fa_idx, sa), (fb_idx, sb) = self.map[e.index]; g0_boundary = None
            is_creased = self.crease_data[e.index] > 0.99
            
            if is_creased:
                fa_is_flat = self.planar_map.get(fa_idx, False); fb_is_flat = self.planar_map.get(fb_idx, False)
                if fa_is_flat and not fb_is_flat: g0_boundary = self._get_b(self.initial[fa_idx], sa)
                elif fb_is_flat and not fa_is_flat: g0_boundary = self._get_b(self.initial[fb_idx], sb, reverse=True)
            if g0_boundary is None:
                pa, pb = self.initial[fa_idx], self.initial[fb_idx]; ca, cb = self._get_b(pa, sa), self._get_b(pb, sb, reverse=True)
                g0_boundary = [(p1 + p2) / 2.0 for p1, p2 in zip(ca, cb)]
            
            cos, coe = Vector(self.verts[e.verts[0].index]), Vector(self.verts[e.verts[1].index])

            if (g0_boundary[0] - cos).length_squared > (g0_boundary[0] - coe).length_squared: cos, coe = coe, cos
            g0_boundary[0], g0_boundary[3] = cos, coe
            self._set_b(self.final[fa_idx], sa, g0_boundary); self._set_b(self.final[fb_idx], sb, g0_boundary, reverse=True)

    def _solve_G1(self):
        if SCIPY_AVAILABLE: self._solve_G1_scipy()
        else: self._solve_G1_numpy()

    def _solve_G1_numpy(self):
        v_map={(f,r,c):i for i,(f,r,c) in enumerate((fi,r,c) for fi in self.initial for r in range(1,3) for c in range(1,3))}
        if not v_map:return
        rows,b_vectors=[],[]
        for e in self.bm.edges:
            if e.is_boundary or len(self.map.get(e.index,[]))!=2 or self.crease_data[e.index] > 0.99: continue
            (fa,sa),(fb,sb)=self.map[e.index];ca,cb,bd=self._get_i(sa),self._get_i(sb,reverse=True),self._get_bi(sa)
            vs,ve=e.verts;is_transition=self.planar_map.get(fa,False)!=self.planar_map.get(fb,False)
            for i in range(4):
                w=1.0;
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
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_x = executor.submit(conjugate_gradient_solver, A, Atb[:, 0])
            future_y = executor.submit(conjugate_gradient_solver, A, Atb[:, 1])
            future_z = executor.submit(conjugate_gradient_solver, A, Atb[:, 2])
            sol_x, sol_y, sol_z = future_x.result(), future_y.result(), future_z.result()
            
        for(f,r,c),i in v_map.items():self.final[f][r][c]=Vector((sol_x[i],sol_y[i],sol_z[i]))

    def _solve_G1_scipy(self):
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import cg

        v_map={(f,r,c):i for i,(f,r,c) in enumerate((fi,r,c) for fi in self.initial for r in range(1,3) for c in range(1,3))}
        if not v_map: return

        rows,b_vectors=[],[]
        for e in self.bm.edges:
            if e.is_boundary or len(self.map.get(e.index,[]))!=2 or self.crease_data[e.index] > 0.99: continue
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

        n=len(v_map)
        A = lil_matrix((n,n))
        b_np=np.array([[vec.x,vec.y,vec.z] for vec in b_vectors])
        Atb=np.zeros((n,3))
        for i,row in enumerate(rows):
            for i1,v1 in row:
                Atb[i1]+=v1*b_np[i]
                for i2,v2 in row: A[i1,i2]+=v1*v2
        
        A_csr = A.tocsr()
        sol_x, _ = cg(A_csr, Atb[:,0])
        sol_y, _ = cg(A_csr, Atb[:,1])
        sol_z, _ = cg(A_csr, Atb[:,2])
        
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
        start_time=time.time()
        backend_msg = "SciPy (High-Performance)" if SCIPY_AVAILABLE else "NumPy (Standard)"
        print(f"Starting NURBS conversion ({bl_info['name']}) using {backend_msg} backend...")
        
        bm_orig = bmesh.new(); bm_orig.from_mesh(self.obj.data)
        deps = bpy.context.evaluated_depsgraph_get(); self.obj_eval = self.obj.evaluated_get(deps)
        bm_eval = self.get_evaluated_bmesh()
        bm_orig.verts.ensure_lookup_table(); bm_orig.faces.ensure_lookup_table(); bm_orig.edges.ensure_lookup_table()

        print("   - Pre-fetching geometry data...")
        num_eval_verts = len(self.obj_eval.data.vertices)
        eval_coords_np = np.empty(num_eval_verts * 3, dtype=np.float32)
        self.obj_eval.data.vertices.foreach_get("co", eval_coords_np)
        eval_coords_np = eval_coords_np.reshape(num_eval_verts, 3)
        
        eval_verts_tuple = tuple(bm_eval.verts)
        
        # --- DEFINITIVE FIX: Due to a Blender API limitation, edge crease data cannot be read in bulk.
        crease_layer = bm_orig.edges.layers.float.get("crease_edge")
        if crease_layer:
            crease_data_np = np.array([e[crease_layer] for e in bm_orig.edges], dtype=np.float32)
        else:
            crease_data_np = np.zeros(len(bm_orig.edges), dtype=np.float32)

        print("Phase 1: Initial patch fitting...")
        self.oriented_fit_cache={};self.ruled_map={};self.fallback_face_indices=set()
        
        for f in bm_orig.faces:
            if len(f.verts)!=4:continue
            
            trace_result=self.trace_face_grid(f, eval_verts_tuple, bm_eval)
            if trace_result is None:
                print(f"   - Path tracer failed for face {f.index}, marking for robust fallback.")
                self.fallback_face_indices.add(f.index)
                trace_result = self.trace_face_grid_projected(f, eval_coords_np)
            if not trace_result:
                print(f"   - All tracers failed for face {f.index}. Skipping.");continue
            
            point_grid_np, orientation, m, n = trace_result
            control_points = self.fit_cps_to_grid(point_grid_np, m, n)

            if not control_points: continue
            self.oriented_fit_cache[f.index]={'cps':control_points,'orientation':orientation}
            if self.settings.generate_ruled_surfaces and self.check_if_ruled_face(f, crease_data_np, self.settings.planar_tolerance):
                self.ruled_map[f.index]=True
        
        if not self.oriented_fit_cache:
            self.report({'ERROR'},"Initial adjustment failure.");bm_orig.free();bm_eval.free();return None
        
        print("Phase 2: Global stitching...")
        temp_stitcher = EdgeCentricStitcher(bm_orig, self.oriented_fit_cache, self.obj, self.settings, {}, self.ruled_map, crease_data_np)
        g0_stitched_patches = temp_stitcher.run(eval_coords_np)
        
        self.planar_map={idx:is_patch_truly_flat(cps,self.settings.planar_tolerance) for idx,cps in g0_stitched_patches.items()}
        
        stitcher = EdgeCentricStitcher(bm_orig, self.oriented_fit_cache, self.obj, self.settings, self.planar_map, self.ruled_map, crease_data_np)
        final_patches = stitcher.run(eval_coords_np)
        
        if self.fallback_face_indices:
            print("Phase 2.1: Applying robust fallback for complex patches...")
            final_patches = self.apply_coons_fallback(final_patches)
        if self.settings.generate_ruled_surfaces and self.ruled_map:
            print("Phase 2.2: Enforcing ruled surface linearity...")
            final_patches=self.enforce_ruled_surfaces(final_patches, bm_orig, crease_data_np)
        if self.settings.force_planar_faces:
            print("Phase 2.5: Forcing planarity...")
            final_patches=self.get_planarized_patches(final_patches)
        
        print(f"Phase 3: Creating objects of type '{self.settings.output_type}'...")
        collection=None
        if self.settings.output_type=='BLENDER_NURBS':collection=self._create_nurbs_objects(final_patches,f"{self.obj.name}_NURBS_Patches")
        elif self.settings.output_type=='PSYCHOPATCH':
            collection=self._create_psychopatch_objects(final_patches,f"{self.obj.name}_Psycho_Patches")
            if not collection:self.report({'ERROR'},"Unable to load ‘NURBS Patch’ from surfacepsycho.")
        
        bm_orig.free();bm_eval.free()
        print(f"✅ NURBS conversion completed in {time.time()-start_time:.2f}s.")
        return collection

    def apply_coons_fallback(self, final_patches):
        corrected_patches = {idx: [[cp.copy() for cp in row] for row in cps] for idx, cps in final_patches.items()}
        fallback_count = 0
        for f_idx in self.fallback_face_indices:
            if f_idx not in corrected_patches: continue
            P = corrected_patches[f_idx]
            P00, P03, P30, P33 = P[0][0], P[0][3], P[3][0], P[3][3]
            P10, P20, P01, P02 = P[1][0], P[2][0], P[0][1], P[0][2]
            P13, P23, P31, P32 = P[1][3], P[2][3], P[3][1], P[3][2]
            L11 = (2/3)*P01 + (1/3)*P31; L12 = (1/3)*P01 + (2/3)*P31; L21 = (2/3)*P02 + (1/3)*P32; L22 = (1/3)*P02 + (2/3)*P32
            Ru11 = (2/3)*P10 + (1/3)*P13; Ru12 = (2/3)*P20 + (1/3)*P23; Ru21 = (1/3)*P10 + (2/3)*P13; Ru22 = (1/3)*P20 + (2/3)*P23
            Rv11 = (2/3)*P01 + (1/3)*P31; Rv12 = (1/3)*P01 + (2/3)*P31; Rv21 = (2/3)*P02 + (1/3)*P32; Rv22 = (1/3)*P02 + (2/3)*P32
            P[1][1] = Ru11 + Rv11 - L11; P[1][2] = Ru11 + Rv21 - L21; P[2][1] = Ru12 + Rv12 - L12; P[2][2] = Ru12 + Rv22 - L22
            fallback_count += 1
        if fallback_count > 0: print(f"     - Rebuilt {fallback_count} complex patches using Coons interpolation.")
        return corrected_patches
    
    def _calculate_grid_length_np(self, grid_np):
        len_h = np.sum(np.linalg.norm(np.diff(grid_np, axis=1), axis=2))
        len_v = np.sum(np.linalg.norm(np.diff(grid_np, axis=0), axis=2))
        return len_h + len_v
    
    def trace_face_grid_projected(self, f, eval_coords_np):
        try:
            v_indices = [l.vert.index for l in f.loops]
            v0e, v1e, v2e, v3e = (Vector(eval_coords_np[i]) for i in v_indices)
        except (KeyError, IndexError): return None
        
        m, n = 8, 8
        u_vals = np.linspace(0, 1, m).reshape(1, m, 1)
        v_vals = np.linspace(0, 1, n).reshape(n, 1, 1)
        v0e, v1e, v2e, v3e = (np.array(v) for v in (v0e, v1e, v2e, v3e))
        
        grid_u_guess = (1-v_vals) * ((1-u_vals) * v0e + u_vals * v1e) + v_vals * ((1-u_vals) * v3e + u_vals * v2e)
        len_u = self._calculate_grid_length_np(grid_u_guess)
        
        grid_v_guess = (1-u_vals) * ((1-v_vals) * v0e + v_vals * v3e) + u_vals * ((1-v_vals) * v1e + v_vals * v2e)
        len_v = self._calculate_grid_length_np(grid_v_guess)

        best_guess_grid, orientation = (grid_u_guess, 0) if len_u <= len_v else (grid_v_guess, 1)
        
        rows, cols = best_guess_grid.shape[:2]
        
        final_grid_rows = []
        for row_idx in range(rows):
            new_row = []
            for col_idx in range(cols):
                p_guess_local = Vector(best_guess_grid[row_idx, col_idx])
                p_guess_world = self.matrix_world @ p_guess_local
                result, location, norm, face_idx = self.obj_eval.closest_point_on_mesh(p_guess_world)
                if result: new_row.append(self.matrix_world.inverted() @ location)
                else: new_row.append(p_guess_local)
            if len(new_row) == cols: final_grid_rows.append(new_row)
        
        if len(final_grid_rows) != rows: return None
        
        final_grid_np = np.array([[v.x, v.y, v.z] for row in final_grid_rows for v in row], dtype=np.float32).reshape(rows, cols, 3)

        if orientation == 1:
            final_grid_np = np.transpose(final_grid_np, (1, 0, 2))
            m, n = n, m
        
        return final_grid_np, orientation, m, n

    def enforce_ruled_surfaces(self, final_patches, bm, crease_data_np):
        print("   - Post-processing ruled surfaces for linearity..."); corrected_patches = {idx: [[cp.copy() for cp in row] for row in cps] for idx, cps in final_patches.items()}; ruled_count = 0
        for face_idx in self.ruled_map:
            if face_idx not in corrected_patches: continue
            patch_cps = corrected_patches[face_idx]; face = bm.faces[face_idx]; o = self.oriented_fit_cache[face_idx]['orientation']
            creases = [crease_data_np[l.edge.index] for l in face.loops]

            is_v_ruled = (o == 0 and creases[0] > 0.99 and creases[2] > 0.99) or (o == 1 and creases[1] > 0.99 and creases[3] > 0.99)
            is_u_ruled = (o == 0 and creases[1] > 0.99 and creases[3] > 0.99) or (o == 1 and creases[0] > 0.99 and creases[2] > 0.99)
            if is_v_ruled:
                ruled_count += 1; P0 = patch_cps[0]; P3 = patch_cps[3]
                for j in range(4): patch_cps[1][j] = (P0[j]*2+P3[j])/3.0; patch_cps[2][j] = (P0[j]+P3[j]*2)/3.0
            elif is_u_ruled:
                ruled_count += 1
                for i in range(4):
                    P_i0=patch_cps[i][0]; P_i3=patch_cps[i][3]; patch_cps[i][1]=(P_i0*2+P_i3)/3.0; patch_cps[i][2]=(P_i0+P_i3*2)/3.0
        if ruled_count > 0: print(f"     - Linearity enforced on {ruled_count} ruled patches."); return corrected_patches
        return corrected_patches
    
    def check_if_ruled_face(self, face, crease_data_np, tolerance=1e-4):
        creases = [crease_data_np[l.edge.index] for l in face.loops]
        crease_sum = sum(1 for c in creases if c > 0.99)
        if crease_sum != 2: return False
        
        smooth_edge1, smooth_edge2 = None, None
        edges = [l.edge for l in face.loops]
        if creases[0] > 0.99 and creases[2] > 0.99: smooth_edge1, smooth_edge2 = edges[1], edges[3]
        elif creases[1] > 0.99 and creases[3] > 0.99: smooth_edge1, smooth_edge2 = edges[0], edges[2]
        else: return False
        
        verts = [v.co for v in face.verts]
        if len(verts)!=4: return False
        p0, p1, p2, p3 = verts[0],verts[1],verts[2],verts[3];plane_normal=(p1-p0).cross(p3-p0)
        if plane_normal.length_squared < 1e-12: return False
        plane_normal.normalize()
        if abs((p2-p0).dot(plane_normal)) > tolerance: return False
        len1 = smooth_edge1.calc_length(); len2 = smooth_edge2.calc_length()
        if len1==0 or len2==0: return False
        if abs(len1-len2)/max(len1,len2)>0.05: return False
        return True

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
        if planar_faces_count > 0: print(f"     - Flatness forcing applied to {planar_faces_count} patches.")
        return new_patches
    
    def create_nurbs_template(self):
        bpy.ops.surface.primitive_nurbs_surface_surface_add(radius=1, location=(9e9, 9e9, 9e9))
        obj = bpy.context.active_object
        data = obj.data.copy()
        spline = data.splines[0]
        spline.use_endpoint_u, spline.use_endpoint_v = True, True
        spline.order_u, spline.order_v = 4, 4
        
        if len(spline.points) < 16:
            spline.points.add(16 - len(spline.points))

        bpy.data.objects.remove(obj)
        return data

    def _create_nurbs_objects(self,patches, col_name):
        nurbs_col=bpy.data.collections.get(col_name) or bpy.data.collections.new(col_name)
        if col_name not in self.context.scene.collection.children:self.context.scene.collection.children.link(nurbs_col)
        for obj in nurbs_col.objects:bpy.data.objects.remove(obj,do_unlink=True)
        created_objects = []
        
        final_coords_np = np.ones((16, 4), dtype=np.float32)

        for face_idx, cps in patches.items():
            name=f"{self.obj.name}_Patch_{face_idx:04d}"
            nurbs_data=self.template_data.copy()
            spline=nurbs_data.splines[0]

            cps_flat = [p.to_tuple() for row in cps for p in row]
            final_coords_np[:, :3] = cps_flat

            spline.points.foreach_set("co", final_coords_np.ravel())

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
            control_points = [cp for row in cps for cp in row]
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
    
    def trace_face_grid(self, f, eval_verts_tuple, bm_e):
        try:
            v_indices = [l.vert.index for l in f.loops]
            v0,v1,v2,v3 = (eval_verts_tuple[i] for i in v_indices)
        except (KeyError,IndexError): return None
        
        result = self._try_build_grid(v0,v3,v1,v2,bm_e)
        if result: 
            grid, m, n = result
            grid_np = np.array([[v.co[i] for i in range(3)] for row in grid for v in row], dtype=np.float32).reshape(m, n, 3)
            return grid_np, 0, m, n
        
        result = self._try_build_grid(v0,v1,v3,v2,bm_e)
        if result: 
            grid, m, n = result
            grid_np = np.array([[v.co[i] for i in range(3)] for row in grid for v in row], dtype=np.float32).reshape(m, n, 3)
            return np.transpose(grid_np, (1, 0, 2)), 1, n, m
        return None
    
    def fit_cps_to_grid(self, point_grid_np, m, n):
        if m==0 or n==0: return None
        num_cps_u,num_cps_v=4,4
        Q = point_grid_np.reshape(m * n, 3)
        def bernstein(n_param,i,t):from math import comb;return comb(n_param,i)*(t**i)*((1-t)**(n_param-i))
        t_u=np.linspace(0,1,m);t_v=np.linspace(0,1,n);B_u=np.zeros((m,num_cps_u));B_v=np.zeros((n,num_cps_v))
        for i in range(num_cps_u):
            for j in range(m): B_u[j,i]=bernstein(num_cps_u-1,i,t_u[j])
        for i in range(num_cps_v):
            for j in range(n): B_v[j,i]=bernstein(num_cps_v-1,i,t_v[j])
        N=np.kron(B_v,B_u)
        try:
            P_flat, _, _, _ = np.linalg.lstsq(N, Q, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return[[Vector(P_flat[r*num_cps_v+c]) for c in range(num_cps_v)] for r in range(num_cps_u)]

# --- Operators (Unchanged) ---
class SUBDIV_OT_convert_to_nurbs(bpy.types.Operator):
    bl_idname="subdiv.convert_to_nurbs";bl_label="Convert SubD to Patches";bl_options={'REGISTER','UNDO'}
    @classmethod
    def poll(cls,context):obj=context.active_object;return obj and obj.type=='MESH' and any(m.type=='SUBSURF' for m in obj.modifiers)
    def execute(self,context):
        obj=context.active_object;settings=context.scene.nurbs_converter_settings
        if settings.output_type == 'PSYCHOPATCH' and settings.psychopatch_extract_normals:
            if not context.scene.render.use_high_quality_normals:
                context.scene.render.use_high_quality_normals = True;self.report({'INFO'}, "Enabled 'High Quality Normals' for PsychoPatch.")
        try:
            fitter=LimitSurfaceFitter(context,obj,settings);fitter.report=self.report
            if(fitter.run_conversion()):self.report({'INFO'},"Conversion complete.")
            else:self.report({'WARNING'},"NURBS conversion failed. Check console for details.")
        except Exception as e:
            self.report({'ERROR'},f"Unexpected error: {e}");import traceback;traceback.print_exc()
            return{'CANCELLED'}
        return{'FINISHED'}
class SUBDIV_OT_toggle_auto_update(bpy.types.Operator):
    bl_idname = "subdiv.toggle_auto_update"; bl_label = "Toggle Auto-Update"; bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        settings = context.scene.nurbs_converter_settings
        if settings.is_auto_update_active:
            ensure_handlers_are_registered(False); clear_auto_update_state(context.scene)
        else:
            active_obj = context.active_object
            if not (active_obj and active_obj.type == 'MESH' and any(m.type == 'SUBSURF' for m in active_obj.modifiers)):
                self.report({'WARNING'}, "Select a mesh with a Subdivision modifier first."); return {'CANCELLED'}
            settings.is_auto_update_active = True
            set_auto_update_source_object(context.scene, active_obj)
            ensure_handlers_are_registered(True)
            bpy.ops.subdiv.convert_to_nurbs('EXEC_DEFAULT')
        return {'FINISHED'}

# --- IGES Exporter (Unchanged) ---
class EXPORT_OT_iges(bpy.types.Operator):
    """Exports selected NURBS and PsychoPatch objects to a single IGES file"""
    bl_idname = "export_scene.subd_to_nurbs_iges"
    bl_label = "Export Patches to IGES"
    bl_options = {'REGISTER'}
    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.igs;*.iges", options={'HIDDEN'})
    def execute(self, context):
        if not self.filepath.lower().endswith(('.igs', '.iges')):
            self.filepath += ".igs"
        selected_objects = [obj for obj in context.selected_objects if obj.type in {'SURFACE', 'MESH'}]
        if not selected_objects:
            self.report({'WARNING'}, "No suitable objects selected for export.")
            return {'CANCELLED'}
        start_lines, global_lines, dir_entry_lines, param_data_lines = [], [], [], []
        de_line_num, pd_line_num = 1, 1
        self._generate_start_section(start_lines)
        self._generate_global_section(global_lines)
        processed_patches_count = 0
        for obj in selected_objects:
            patch_data = self._extract_patch_data(obj)
            if not patch_data: continue
            de_lines, pd_lines, de_line_num, pd_line_num = self._process_patch(
                patch_data, de_line_num, pd_line_num
            )
            dir_entry_lines.extend(de_lines)
            param_data_lines.extend(pd_lines)
            processed_patches_count += 1
        if processed_patches_count == 0:
            self.report({'WARNING'}, "No valid patch objects were found to export.")
            return {'CANCELLED'}
        try:
            with open(self.filepath, 'w', encoding='ascii') as f:
                f.writelines(start_lines); f.writelines(global_lines)
                f.writelines(dir_entry_lines); f.writelines(param_data_lines)
                term_str = f"S{len(start_lines):>7}G{len(global_lines):>7}D{len(dir_entry_lines):>7}P{len(param_data_lines):>7}"
                f.write(self._format_line('T', term_str, 1))
        except IOError as e:
            self.report({'ERROR'}, f"Failed to write file: {str(e)}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Successfully exported {processed_patches_count} patches to {self.filepath}")
        return {'FINISHED'}
    def invoke(self, context, event):
        blend_filename = Path(bpy.data.filepath).stem
        self.filepath = f"{blend_filename}.igs" if blend_filename else "Untitled.igs"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    def _format_line(self, section_char, content, line_num):
        return f"{content.ljust(72)}{section_char}{line_num:>7}\n"
    def _generate_start_section(self, buffer):
        buffer.append(self._format_line('S', f"Exported from Blender using SubD to NURBS Addon v{'.'.join(map(str, bl_info['version']))}", 1))
    def _generate_global_section(self, buffer):
        now = datetime.datetime.now().strftime("%Y%m%d.%H%M%S")
        filepath_ascii = self.filepath.encode('ascii', 'replace').decode('ascii')
        filepath_str = f"{len(filepath_ascii)}H{filepath_ascii}"
        params = [
            "1H,", "1H;", f"{filepath_str}", "31HBlender (SubD to NURBS Addon)", "", "16H", "32", "308",
            "15", "1.0", "3", "7HMM", "1.0", f"16H{now}", "0.0001", "10000.0", "", "2H", "", ""
        ]
        global_str = ",".join(params) + ";"
        for i, chunk in enumerate((global_str[i:i+72] for i in range(0, len(global_str), 72))):
            buffer.append(self._format_line('G', chunk, i + 1))
    def _generate_clamped_knot_vector(self, order, num_points):
        return [0.0] * order + [1.0] * order
    def _extract_patch_data(self, obj):
        matrix = obj.matrix_world.copy()
        if obj.type == 'SURFACE' and obj.data.splines and obj.data.splines[0].type == 'NURBS':
            spline = obj.data.splines[0]
            num_u, num_v = spline.point_count_u, spline.point_count_v
            order_u, order_v = spline.order_u, spline.order_v
            
            num_points = len(spline.points)
            coords_flat = np.empty(num_points * 4, dtype=np.float32)
            spline.points.foreach_get("co", coords_flat)
            coords_np = coords_flat.reshape(num_points, 4)
            
            points_3d = coords_np[:, :3]
            points_hom = np.hstack((points_3d, np.ones((num_points, 1), dtype=np.float32)))
            transformed_points = (points_hom @ np.array(matrix).T)[:, :3] * 1000.0
            
            cps_flat = transformed_points.ravel().tolist()
            weights_flat = coords_np[:, 3].tolist()

            return {"name": obj.name, "num_u": num_u, "num_v": num_v, "order_u": order_u, "order_v": order_v, "cps": cps_flat, "weights": weights_flat}
        elif obj.type == 'MESH' and len(obj.data.vertices) == 16:
            num_u, num_v, order_u, order_v = 4, 4, 4, 4

            coords_flat = np.empty(16 * 3, dtype=np.float32)
            obj.data.vertices.foreach_get("co", coords_flat)
            coords_np = coords_flat.reshape(16, 3)
            
            points_hom = np.hstack((coords_np, np.ones((16, 1), dtype=np.float32)))
            transformed_points = (points_hom @ np.array(matrix).T)[:, :3] * 1000.0
            cps_flat = transformed_points.ravel().tolist()

            return {"name": obj.name, "num_u": num_u, "num_v": num_v, "order_u": order_u, "order_v": order_v, "cps": cps_flat, "weights": [1.0] * 16}
        return None
    def _process_patch(self, data, start_de_num, start_pd_num):
        de_lines, pd_lines = [], []
        de_ptr_val = start_de_num
        degree_u, degree_v = data['order_u'] - 1, data['order_v'] - 1
        knots_u = self._generate_clamped_knot_vector(data['order_u'], data['num_u'])
        knots_v = self._generate_clamped_knot_vector(data['order_v'], data['num_v'])
        params = ["128", data['num_u'] - 1, data['num_v'] - 1, degree_u, degree_v, "0", "0", "1", "0", "0"]
        params.extend(knots_u); params.extend(knots_v)
        params.extend(data['weights']); params.extend(data['cps'])
        params.extend([0.0, 1.0, 0.0, 1.0])
        current_line, current_pd_num = "", start_pd_num
        for param in params:
            param_str = f"{param:.7G}," if isinstance(param, float) else f"{param},"
            if len(current_line) + len(param_str) > 64:
                content = f"{current_line.ljust(64)}{de_ptr_val:>8}"
                pd_lines.append(self._format_line('P', content, current_pd_num))
                current_pd_num += 1; current_line = ""
            current_line += param_str
        final_line_str = current_line[:-1] + ";"
        content = f"{final_line_str.ljust(64)}{de_ptr_val:>8}"
        pd_lines.append(self._format_line('P', content, current_pd_num))
        num_param_lines = len(pd_lines)
        line1_data = (f"{128:>8}" f"{start_pd_num:>8}" f"{0:>8}" f"{0:>8}" f"{0:>8}" f"{0:>8}" f"{0:>8}" f"{0:>8}" f"{'00000001':>8}")
        de_lines.append(self._format_line('D', line1_data, start_de_num))
        line2_data = (f"{128:>8}" f"{0:>8}" f"{0:>8}" f"{num_param_lines:>8}" f"{1:>8}" f"{0:>8}" f"{0:>8}" f"{data['name'].ljust(8)[:8]:<8}" f"{0:>8}")
        de_lines.append(self._format_line('D', line2_data, start_de_num + 1))
        return de_lines, pd_lines, start_de_num + 2, current_pd_num + 1

# --- Dependency Management UI (Unchanged) ---
class SUBDIV_OT_install_dependencies(bpy.types.Operator):
    """Installs the SciPy library for a massive performance boost."""
    bl_idname = "subdiv.install_dependencies"
    bl_label = "Install SciPy Dependency"
    bl_description = "Downloads and installs SciPy into Blender's user scripts folder. This enables a much faster algorithm for patch stitching"
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    def execute(self, context):
        self.report({'INFO'}, "Starting SciPy installation. This may take a few minutes...")
        python_exe = sys.executable; target_path = get_dependency_path()
        try:
            subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True)
            subprocess.run([python_exe, "-m", "pip", "install", f"--target={target_path}", "scipy"], check=True, capture_output=True)
            check_scipy()
            self.report({'INFO'}, "SciPy installed successfully! PLEASE RESTART BLENDER to activate the new backend.")
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            self.report({'ERROR'}, f"SciPy installation failed. Check Blender's system console for details. Error: {err}")
            return {'CANCELLED'}
        return {'FINISHED'}

class SubDtoNurbsAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    def draw(self, context):
        layout = self.layout; box = layout.box()
        box.label(text="Performance Backend", icon='SETTINGS')
        if SCIPY_AVAILABLE:
            row = box.row(align=True)
            row.label(text="SciPy Backend: Active", icon='CHECKMARK')
            row.label(text="(Highest Performance)")
        else:
            box.label(text="The high-performance SciPy backend is not installed.")
            box.label(text="The addon is currently using the standard NumPy backend.")
            box.separator()
            box.label(text="For a significant speed-up on complex models, please install SciPy.")
            box.operator(SUBDIV_OT_install_dependencies.bl_idname, icon='CONSOLE')

# --- UI Panels (Unchanged) ---
class SUBDIV_PT_converter_panel(bpy.types.Panel):
    bl_label="SubD to NURBS Converter";bl_idname="SUBDIV_PT_converter_panel";bl_space_type='VIEW_3D';bl_region_type='UI';bl_category="SubD to NURBS"
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and any(m.type == 'SUBSURF' for m in obj.modifiers)
    def draw(self,context):
        layout=self.layout;obj=context.active_object;settings=context.scene.nurbs_converter_settings
        layout.label(text=f"Source: {obj.name}",icon='MESH_DATA')
        mod = next(m for m in obj.modifiers if m.type == 'SUBSURF')
        box=layout.box();box.prop(mod,"levels",text="Subdivision Level")
        settings=context.scene.nurbs_converter_settings;box=layout.box()
        box.label(text="Output Type");box.prop(settings,"output_type",expand=True)
        post_box=layout.box();post_box.label(text="Post-process:")
        col=post_box.column(align=True);col.prop(settings,"output_material")
        if settings.output_type=='BLENDER_NURBS':
            col.prop(settings,"join_patches")
            sub_col=col.column(align=True);sub_col.active=settings.join_patches
            sub_col.prop(settings,"add_weld_modifier");sub_col.prop(settings,"add_subd_modifier");sub_col.prop(settings,"add_smooth_modifier")
        elif settings.output_type=='PSYCHOPATCH':
            col.prop(settings,"psychopatch_extract_normals")
        layout.separator()
        layout.operator("subdiv.convert_to_nurbs",text="Convert to Patches",icon='SURFACE_DATA')
        layout.separator()
        box = layout.box()
        if settings.is_auto_update_active:
            source_obj = get_auto_update_source_object(context)
            if source_obj and source_obj.name == obj.name:
                box.label(text=f"Auto-update ON for: {source_obj.name}", icon='INFO')
                box.operator("subdiv.toggle_auto_update", text="Stop Auto-Update", icon='PAUSE')
            else:
                if source_obj: box.label(text=f"Watching: {source_obj.name}", icon='INFO')
                box.label(text="Stop from original object's panel.", icon='ERROR')
        else:
            box.operator("subdiv.toggle_auto_update", text="Start Auto-Update", icon='PLAY')
        layout.separator()
        adv_box=layout.box();adv_box.label(text="Correction Parameters",icon='SETTINGS')
        col=adv_box.column(align=True);col.prop(settings,"generate_ruled_surfaces",text="Generate Ruled Surfaces")
        row=col.row(align=True);row.prop(settings,"force_planar_faces")
        sub_row=row.row(align=True);sub_row.active=settings.force_planar_faces;sub_row.prop(settings,"planar_tolerance",text="Threshold")
        col.prop(settings,"g1_transition_weight");col.prop(settings,"exceptional_weight");col.prop(settings,"fairing_weight")

class NURBS_PT_tools_panel(bpy.types.Panel):
    bl_label = "NURBS Tools"; bl_idname = "NURBS_PT_tools_panel"; bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'; bl_category = "SubD to NURBS"
    @classmethod
    def poll(cls, context):
        def is_valid_export_object(obj):
            if obj.type == 'SURFACE': return True
            if obj.type == 'MESH':
                for mod in obj.modifiers:
                    if mod.type == 'NODES' and mod.name == "SP - NURBS Patch Meshing":
                        return True
            return False
        return any(is_valid_export_object(obj) for obj in context.selected_objects)
    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Export Tools:")
        box.operator(EXPORT_OT_iges.bl_idname, text="Export Selected as IGES", icon='EXPORT')

# --- Registration ---
classes_to_register = (
    NurbsConverterSettings,
    SUBDIV_OT_convert_to_nurbs,
    SUBDIV_OT_toggle_auto_update,
    SUBDIV_OT_install_dependencies,
    SubDtoNurbsAddonPreferences,
    EXPORT_OT_iges,
    SUBDIV_PT_converter_panel,
    NURBS_PT_tools_panel,
)

def register():
    ensure_dependency_path()
    check_scipy()
    for cls in classes_to_register:
        bpy.utils.register_class(cls)
    bpy.types.Scene.nurbs_converter_settings = bpy.props.PointerProperty(type=NurbsConverterSettings)
    ensure_handlers_are_registered(False)

def unregister():
    ensure_handlers_are_registered(False)
    try:
        if bpy.context and bpy.context.scene: clear_auto_update_state(bpy.context.scene)
    except (AttributeError, RuntimeError): pass
    for cls in reversed(classes_to_register):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, 'nurbs_converter_settings'):
        del bpy.types.Scene.nurbs_converter_settings

if __name__=="__main__":
    register()