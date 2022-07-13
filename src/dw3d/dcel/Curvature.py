#Sacha Ichbiah, Sept 2021
import numpy as np 
import scipy.sparse as sp
import robust_laplacian

def find_key_multiplier(num_points): 
    key_multiplier = 1
    while num_points//key_multiplier != 0 : 
        key_multiplier*=10
    return(key_multiplier)   

def compute_laplacian_cotan(Mesh): 
    ### Traditional cotan laplacian : from
    # from "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", Meyer et al. 2003

    #Implementation and following explanation from : pytorch3d/loss/mesh_laplacian_smoothing.py

    r"""
    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN matrix such that LV gives a matrix of vectors:
    LV[i] gives the normal scaled by the discrete mean curvature. 
    For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.
    .. code-block:: python
               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij
        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.
    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.
    .. code-block:: python
               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C
        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have
        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a
        Putting these together, we get:
        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH
    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.
    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, inv_areas=laplacian_cot(verts,faces)
    inv_areas = inv_areas.reshape(-1)
    sum_cols = np.array(L.sum(axis=1))
    Laplacian = L@verts - verts*sum_cols
    norm = (0.75*inv_areas).reshape(-1,1)
    return(Laplacian*norm,inv_areas)

def compute_laplacian_robust(Mesh): 
    ### Robust Laplacian using implicit triangulations : 
    # from "A Laplacian for Nonmanifold Triangle Meshes", N.Sharp, K.Crane, 2020
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = 1/M.diagonal().reshape(-1)/3
    Sum_cols = np.array(L.sum(axis=1)) #Useless as it is already 0 (sum comprised in the central term) see http://rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)
    norm = (1.5*inv_areas).reshape(-1,1)
    return(-Laplacian*norm,inv_areas)

def compute_curvature_vertices_cotan(Mesh): 
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, inv_areas=laplacian_cot(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = inv_areas.reshape(-1)
    Sum_cols = np.array(L.sum(axis=1))
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)/2
    H = np.linalg.norm(Laplacian,axis=1)*3*inv_areas/2
    return(H,inv_areas,Laplacian*3*(np.array([inv_areas]*3).transpose())/2)

def compute_curvature_vertices_robust_laplacian(Mesh): 
    verts = Mesh.v
    faces = Mesh.f[:,[0,1,2]]
    L, M=robust_laplacian.mesh_laplacian(Mesh.v,Mesh.f[:,[0,1,2]])
    inv_areas = 1/M.diagonal().reshape(-1)/3
    Sum_cols = np.array(L.sum(axis=1))
    first_term = np.dot(L.toarray(),verts)
    second_term = verts*Sum_cols
    Laplacian = (first_term-second_term)
    H = np.linalg.norm(Laplacian,axis=1)*3*inv_areas/2
    return(H,inv_areas,Laplacian*3*(np.array([inv_areas]*3).transpose())/2)


def compute_curvature_interfaces(Mesh,laplacian = "robust",weighted=True): 
    Interfaces={}
    Interfaces_weights={}
    if laplacian =="robust" : 
        L,inv_areas = compute_laplacian_robust(Mesh)
    elif laplacian =="cotan" : 
        L,inv_areas = compute_laplacian_cotan(Mesh)

    vertex_normals = Mesh.compute_vertex_normals()
    H = np.sign(np.sum(np.multiply(L,vertex_normals),axis=1))*np.linalg.norm(L,axis=1)
    
    
    Vertices_on_interfaces ={}
    for edge in Mesh.half_edges : 
        #pass trijunctions
        
        materials = (edge.incident_face.material_1,edge.incident_face.material_2)
        interface_key = (min(materials),max(materials))
        
        Vertices_on_interfaces[interface_key]=Vertices_on_interfaces.get(interface_key,[])
        Vertices_on_interfaces[interface_key].append(edge.origin.key)
        Vertices_on_interfaces[interface_key].append(edge.destination.key)
        
    verts_idx = {}
    for key in Vertices_on_interfaces.keys() : 
        verts_idx[key] = np.unique(np.array(Vertices_on_interfaces[key]))
        
    Interfaces_curvatures = {}
    for key in verts_idx.keys() : 
        curvature = 0
        weights = 0
        for vert_idx in verts_idx[key]: 
            v = Mesh.vertices[vert_idx]
            if v.on_trijunction : 
                continue
            else : 
                if weighted : 
                    curvature += H[vert_idx]/inv_areas[vert_idx]
                    weights += 1/inv_areas[vert_idx]
                else : 
                    curvature += H[vert_idx]
                    weights +=1

        """
        TEMPORARY : 
        FOR THE MOMENT, WE CANNOT COMPUTE CURVATURE ON LITTLE INTERFACES
        THERE ARE THREE POSSIBILITIES : 
        -WE REFINE THE SURFACES UNTIL THERE IS A VERTEX ON THE SURFACE AND THUS IT CAN BE COMPUTED> 
        -WE PUT THE CURVATURE TO ZERO 
        -WE REMOVE THE EQUATION FROM THE SET OF EQUATIONS

        -Removing the equations could be dangerous as we do not know what we are going to get : 
        maybe the system will become underdetermined, and thus unstable ? 
            -> Unprobable as the systems are strongly overdetermined. 
            -> Bayesian ? 
        -Putting the curvature to zero should not have a strong influence on the inference as in any way during the least-squares
        minimization each equation for the pressures are proportionnal to the area of the interfaces. 
        Thus we return to the case 1, where in fact it does not matter so much if the equation is removed or kept for the little interfaces
        
        -It is thus useless to refine the surface until a curvature can be computed, for sure. 
        
        
        """
        if weights==0 : 
            Interfaces_curvatures[key]=np.nan
        else : 
            Interfaces_curvatures[key]=curvature/weights
    
    return(Interfaces_curvatures)


def cot(x):
    return(1/np.tan(x))
  


def laplacian_cot(verts,faces):
    ##


    """
    Returns the Laplacian matrix with cotangent weights and the inverse of the
    face areas.
    Args:
        meshes: Meshes object with a batch of meshes.
    Returns:
        2-element tuple containing
        - **L**: FloatTensor of shape (V,V) for the Laplacian matrix (V = sum(V_n))
           Here, L[i, j] = cot a_ij + cot b_ij iff (i, j) is an edge in meshes.
           See the description above for more clarity.
        - **inv_areas**: FloatTensor of shape (V,) containing the inverse of sum of
           face areas containing each vertex
    """

    V, F = len(verts),len(faces)

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = np.linalg.norm((v1 - v2),axis=1)
    B = np.linalg.norm((v0 - v2),axis=1)
    C = np.linalg.norm((v0 - v1),axis=1)
    
    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = np.sqrt((s * (s - A) * (s - B) * (s - C)))#.clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = np.stack([cota, cotb, cotc], axis=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = np.stack([ii,jj],axis=0).reshape(2, F*3)
    
    L = sp.coo_matrix((cot.reshape(-1),(idx[1],idx[0])), shape = (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.transpose()

   
    # For each vertex, compute the sum of areas for triangles containing it.
    inv_areas=np.zeros(V)
    idx = faces.reshape(-1)
    val = np.stack([area] * 3, axis=1).reshape(-1)
    np.add.at(inv_areas,idx,val)
    idx = inv_areas > 0
    inv_areas[idx] = 1.0 / inv_areas[idx]
    inv_areas = inv_areas.reshape(-1, 1)

    return L,inv_areas



def compute_areas(faces,verts): 
    Pos = verts[faces]
    Sides = Pos-Pos[:,[2,0,1]]
    Lengths_sides = np.linalg.norm(Sides,axis = 2)
    Half_perimeters = np.sum(Lengths_sides,axis=1)/2

    Diffs = np.array([Half_perimeters]*3).transpose() - Lengths_sides
    Areas = (Half_perimeters*Diffs[:,0]*Diffs[:,1]*Diffs[:,2])**(0.5)
    return(Areas)



def compute_areas_interfaces(Mesh): 
    ###
    #Duplicate of a function present in Geometry (with the same name), but computed in a different manner
    ###

    f = Mesh.f
    v = Mesh.v
    Areas = compute_areas(f[:,[0,1,2]],v)
    Interfaces_areas = {}
    for i, face in enumerate(f): 
        _,_,_,a,b = face
        Table = Interfaces_areas.get((a,b),0)
        Interfaces_areas[(a,b)]=Table+Areas[i]
    return(Interfaces_areas)










"""
HUGE FUNCTION
"""
def compute_curvature_interfaces_cotan(Faces,Verts) : 

    #5 times faster than the iterative version. More principled. 

    dict_curvatures = {}
    #We start from an oriented mesh
    Faces[:,[0,1,2]]=np.sort(Faces[:,[0,1,2]],axis=1)
    Faces[:,[3,4]]=np.sort(Faces[:,[3,4]],axis=1)
    Faces_raw = Faces[:,[0,1,2]]

    Edges = np.vstack((Faces[:,[0,1,3,4]],Faces[:,[1,2,3,4]],Faces[:,[2,0,3,4]]))
    key_mult = find_key_multiplier(np.amax(Edges))
    Keys = Edges[:,0]*key_mult + Edges[:,1]
    Mapping = dict(zip(Keys,np.arange(len(Keys))))

    Instances_edges = {}
    Occupancy = np.zeros(len(Keys))
    for i,key in enumerate(Keys):
        if Occupancy[Mapping[key]]==0 : 
            Instances_edges[key]=set(Edges[i][2:])
            Occupancy[Mapping[key]]=1
        else : 
            Instances_edges[key]=Instances_edges[key].union(set(Edges[i][2:]))

    Edges_implied_in_trijunctions_index = []  
    for key in Instances_edges : 
        if len(Instances_edges[key])>2 : 
            Edges_implied_in_trijunctions_index.append(Mapping[key])

    Verts_implied_in_trijunctions = np.unique(Edges[Edges_implied_in_trijunctions_index][:,[0,1]])
    Table_vert_implied_in_trijunction = np.zeros(len(Verts))
    Table_vert_implied_in_trijunction[Verts_implied_in_trijunctions]=1


    key_mult = find_key_multiplier(np.amax(Faces[:,[3,4]]))
    Keys = Faces[:,3]*key_mult + Faces[:,4]
    keys, index = np.unique(Keys,return_index=True)
    Interfaces = Faces[:,[3,4]][index]
    #Faces with interface_key instead of a pair of numbers
    Faces_with_interface_key = np.hstack((Faces[:,[0,1,2]],Keys.reshape(-1,1)))



    L, inv_areas=laplacian_cot(Verts,Faces[:,[0,1,2]])
    Sum_cols = np.array(L.sum(axis=1))
    first_term = L@Verts
    second_term = Verts*Sum_cols
    Laplacian = (first_term-second_term)/2

    idx_0 = np.array([np.arange(len(Faces))]*3).flatten()
    idx_1 = Faces[:,:3].transpose().flatten()
    M = sp.coo_matrix((np.ones(len(idx_0)),(idx_1,idx_0)),shape=(len(Verts),len(Faces)))
    Faces_areas = compute_areas(Faces[:,:3],Verts)
    Vertex_areas = M@Faces_areas
    Vertex_areas[Vertex_areas==0]+=1
    H = np.linalg.norm(Laplacian,axis=1)/(2)/(Vertex_areas/3)
    for i,key in enumerate(keys) : 
        faces_of_interface = Faces_raw[Faces_with_interface_key[:,3]==key]
        verts_of_interface = np.unique(faces_of_interface)
        verts_considered = verts_of_interface[Table_vert_implied_in_trijunction[verts_of_interface]==0]
        if len(verts_considered)>0 :
            H_c = H[verts_considered]       
            vertex_area_c = Vertex_areas[verts_considered]
            #print(,np.mean(H[verts_considered]),np.sum(H_c*vertex_area_c)/np.sum(vertex_area_c))
            mean_H = np.sum(H_c*vertex_area_c)/np.sum(vertex_area_c)
            dict_curvatures[tuple(Interfaces[i])]=mean_H
            
    return(dict_curvatures)


def compute_curvature_interfaces_iteration(Faces,Verts):

    
    ##important : we do not do the mean of H but the weighted man because otherwise we end up with something too sensitive

    dict_curvatures = {}
    #We sort the faces labels ([v1,v2,v3,a,b] with a<b and v1<v2<v3)
    Faces[:,[0,1,2]]=np.sort(Faces[:,[0,1,2]],axis=1)
    Faces[:,[3,4]]=np.sort(Faces[:,[3,4]],axis=1)
    
    Edges = np.vstack((Faces[:,[0,1,3,4]],Faces[:,[1,2,3,4]],Faces[:,[2,0,3,4]]))
    key_mult = find_key_multiplier(np.amax(Edges))
    Keys = Edges[:,0]*key_mult + Edges[:,1]
    Mapping = dict(zip(Keys,np.arange(len(Keys))))
    
    Instances_edges = {}
    Occupancy = np.zeros(len(Keys))


    for i,key in enumerate(Keys):
        if Occupancy[Mapping[key]]==0 : 
            Instances_edges[key]=set(Edges[i][2:])
            Occupancy[Mapping[key]]=1
        else : 
            Instances_edges[key]=Instances_edges[key].union(set(Edges[i][2:]))

    Edges_implied_in_trijunctions_index = []  
    for key in Instances_edges : 
        if len(Instances_edges[key])>2 : 
            Edges_implied_in_trijunctions_index.append(Mapping[key])

    Verts_implied_in_trijunctions = np.unique(Edges[Edges_implied_in_trijunctions_index][:,[0,1]])
    Table_vert_implied_in_trijunction = np.zeros(len(Verts))
    Table_vert_implied_in_trijunction[Verts_implied_in_trijunctions]=1

    
    #We find on which interface each face is located
    key_mult = find_key_multiplier(np.amax(Faces[:,[3,4]]))
    Keys = Faces[:,3]*key_mult + Faces[:,4]
    keys, index = np.unique(Keys,return_index=True)
    Interfaces = Faces[:,[3,4]][index]
    #Faces with interface_key instead of a pair of numbers
    Faces_with_interface_key = np.hstack((Faces[:,[0,1,2]],Keys.reshape(-1,1)))
    #Area of each face
    Faces_area = compute_areas(Faces[:,[0,1,2]],Verts)


    Occupancy = np.zeros(len(Verts))
    dict_faces_at_each_vertex={}
    for i,face in enumerate(Faces) : 
        for index in face[:3] : 
            if Occupancy[index]==0 : 
                dict_faces_at_each_vertex[index]=[i]
                Occupancy[index]=1
            else : 
                dict_faces_at_each_vertex[index].append(i)

    Faces_raw = Faces[:,[0,1,2]]
    Pos = Verts[Faces_raw]
    Sides = Pos-Pos[:,[1,2,0]]
    Norm = np.linalg.norm(Sides,axis=2)
    Sides/=np.array([Norm]*3).transpose(1,2,0)

    Angle_0 = np.arccos(np.sum(np.multiply(Sides[:,0],Sides[:,2]),axis=1))
    Angle_1 = np.arccos(np.sum(np.multiply(Sides[:,0],Sides[:,1]),axis=1))
    Angle_2 = np.arccos(np.sum(np.multiply(Sides[:,1],Sides[:,2]),axis=1))
    Angles = np.stack([Angle_0,Angle_1,Angle_2]).transpose()


    dict_curvatures = {}
    for i,key in enumerate(keys) :
        interface = Interfaces[i]
        faces_of_interface = Faces_raw[Faces_with_interface_key[:,3]==key]
        verts_of_interface = np.unique(faces_of_interface)
        verts_considered = verts_of_interface[Table_vert_implied_in_trijunction[verts_of_interface]==0]
        
        Vertex_areas = np.zeros(len(verts_considered))
        Laplacians_vertices_considered = np.zeros((len(verts_considered),3))

        for i,vert in enumerate(verts_considered) : 
            # Maintenant on veut relier les faces au vert considéré
            faces_idx = dict_faces_at_each_vertex[vert]
            Vertex_areas[i] = np.sum(Faces_area[faces_idx])

            ### COTAN FORMULA
            Laplacian=0
            for face_idx in faces_idx : 
                face = Faces_raw[face_idx]
                angles = Angles[face_idx]
                a,b,c=face
                Oa,Ob,Oc = angles

                if a == vert : 
                    Laplacian +=cot(Ob)*(Verts[c]-Verts[vert])
                    Laplacian +=cot(Oc)*(Verts[b]-Verts[vert])

                if b == vert : 
                    Laplacian +=cot(Oa)*(Verts[c]-Verts[vert])
                    Laplacian +=cot(Oc)*(Verts[a]-Verts[vert])

                if c == vert : 
                    Laplacian +=cot(Oa)*(Verts[b]-Verts[vert])
                    Laplacian +=cot(Ob)*(Verts[a]-Verts[vert])

            Laplacians_vertices_considered[i]=Laplacian.copy()
        Laplacians_vertices_considered/=2
        H = (np.linalg.norm(Laplacians_vertices_considered,axis=1)/2)/(Vertex_areas/3)
        if len(H)>0 :     
            mean_H = np.sum(H*Vertex_areas)/(np.sum(Vertex_areas))
            dict_curvatures[tuple(interface)]=mean_H
    return(dict_curvatures)









"""
def compute_curvature_interfaces(Mesh,laplacian = "robust"): 
    Interfaces={}
    Interfaces_n_elemts={}
    if laplacian =="robust" : 
        H = compute_curvature_vertices_robust_laplacian(Mesh)
    elif laplacian =="cotan" : 
        H = compute_curvature_vertices_cotan(Mesh)
    for edge in Mesh.half_edges : 
        #pass trijunctions
        if len(edge.twin)>1 : 
            continue
        materials = (edge.incident_face.material_1,edge.incident_face.material_2)
        interface_key = (min(materials),max(materials))
        curvature = H[edge.origin.key]+H[edge.destination.key]
        Interfaces[interface_key]=Interfaces.get(interface_key,0)+curvature
        Interfaces_n_elemts[interface_key]=Interfaces_n_elemts.get(interface_key,0)+2
    Interfaces_curvatures_mean = {}
    for key in Interfaces:
        Interfaces_curvatures_mean[key]=Interfaces[key]/Interfaces_n_elemts[key]
    return(Interfaces_curvatures_mean)
#compute_curvature_interfaces(Mesh): """
"""
def compute_curvature_interfaces(Mesh,laplacian = "robust",weighted=True): 
    Interfaces={}
    Interfaces_weights={}
    if laplacian =="robust" : 
        H,inv_areas,_ = compute_curvature_vertices_robust_laplacian(Mesh)
    elif laplacian =="cotan" : 
        H,inv_areas,_ = compute_curvature_vertices_cotan(Mesh)
    for edge in Mesh.half_edges : 
        #pass trijunctions
        if len(edge.twin)>1 : 
            continue
        materials = (edge.incident_face.material_1,edge.incident_face.material_2)
        interface_key = (min(materials),max(materials))
        
        if weighted: 
            curvature = H[edge.origin.key]/inv_areas[edge.origin.key]
            curvature+= H[edge.destination.key]/inv_areas[edge.destination.key]

            Interfaces[interface_key]=Interfaces.get(interface_key,0)+curvature
            Interfaces_weights[interface_key]=Interfaces_weights.get(interface_key,0)+ 1/inv_areas[edge.origin.key] + 1/inv_areas[edge.destination.key]
        else : 
            curvature = H[edge.origin.key]+H[edge.destination.key]
            Interfaces[interface_key]=Interfaces.get(interface_key,0)+curvature
            Interfaces_weights[interface_key]=Interfaces_weights.get(interface_key,0)+ 2
            
    Interfaces_curvatures_mean = {}
    for key in Interfaces:
        Interfaces_curvatures_mean[key]=Interfaces[key]/Interfaces_weights[key]
    return(Interfaces_curvatures_mean)


def compute_curvature_vectors_interfaces(Mesh,laplacian = "robust",weighted=True): 
    Interfaces={}
    Interfaces_weights={}
    if laplacian =="robust" : 
        H,inv_areas,L = compute_curvature_vertices_robust_laplacian(Mesh)
    elif laplacian =="cotan" : 
        H,inv_areas,L = compute_curvature_vertices_cotan(Mesh)

    for edge in Mesh.half_edges : 
        #pass trijunctions
        if len(edge.twin)>1 : 
            continue
        materials = (edge.incident_face.material_1,edge.incident_face.material_2)
        interface_key = (min(materials),max(materials))
        
        if weighted: 
            curvature_vector = L[edge.origin.key]/inv_areas[edge.origin.key]
            curvature_vector+= L[edge.destination.key]/inv_areas[edge.destination.key]

            Interfaces[interface_key]=Interfaces.get(interface_key,0)+curvature_vector
            Interfaces_weights[interface_key]=Interfaces_weights.get(interface_key,0)+ 1/inv_areas[edge.origin.key] + 1/inv_areas[edge.destination.key]
        else : 
            curvature_vector = L[edge.origin.key]+L[edge.destination.key]
            Interfaces[interface_key]=Interfaces.get(interface_key,0)+curvature_vector
            Interfaces_weights[interface_key]=Interfaces_weights.get(interface_key,0)+ 2
            
    Interfaces_curvatures_mean = {}
    for key in Interfaces:
        Interfaces_curvatures_mean[key]=Interfaces[key]/Interfaces_weights[key]
    return(Interfaces_curvatures_mean)

"""
