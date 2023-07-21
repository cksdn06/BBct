from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance

def ray_tracing(x, y):
	# screen is on origin 
	pixel = np.array([x, y, 0]) 
	origin = camera 
	direction = normalize(pixel - origin) 
	color = np.zeros((3)) 
	reflection = 1 
	for k in range(max_depth): 
		# check for intersections 
		nearest_object, min_distance = nearest_intersected_object(objects, origin, direction) 
		if nearest_object is None: 
			break 
		intersection = origin + min_distance * direction 
		normal_to_surface = normalize(intersection - nearest_object['center']) 
		shifted_point = intersection + 1e-5 * normal_to_surface 
		intersection_to_light = normalize(light['position'] - shifted_point) 
		_, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) 
		intersection_to_light_distance = np.linalg.norm(light['position'] - intersection) 
		is_shadowed = min_distance < intersection_to_light_distance 
		if is_shadowed: 
			break 
		illumination = np.zeros((3)) 
		# ambiant 
		illumination += nearest_object['ambient'] * light['ambient'] 
		# diffuse 
		illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) 
		# specular 
		intersection_to_camera = normalize(camera - intersection) 
		H = normalize(intersection_to_light + intersection_to_camera) 
		illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) 
		# reflection 
		color += reflection * illumination 
		reflection *= nearest_object['reflection'] 
		origin = shifted_point 
		direction = reflected(direction, normal_to_surface)
	return color

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_time = MPI.Wtime()

max_depth = 3

#### parameters
width = 300
height = 200
camera = np.array([0, 0, 1])
#camera = np.array([0, 1, 1])
light = { 'position': np.array([5, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([-0.2, 0, -1]), 'radius': 0.2, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 1, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 80, 'reflection': 0.1 },
    { 'center': np.array([0.1, -0.3, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0.5, 0, -1]), 'radius': 0.5, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.3, 0, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]

# Solar System
'''light = { 'position': np.array([-0.08, 0, 0.33]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }
objects = [
    { 'center': np.array([0, 0, 0]), 'radius': 0.12, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0.7, 0.1]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([-0.6748, -0.1539, 0]), 'radius': 0.02, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'center': np.array([ -0.5868, -0.126, 0]), 'radius': 0.045, 'ambient': np.array([0, 0.1, 0.1]), 'diffuse': np.array([0.1, 0.7, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
   ]'''

ratio = float(width) / height 
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom


if height % size > rank:
    N = 1
else :
    N = 0

st = rank*(height//size) + comm.scan(N,MPI.SUM) - N
ed = (rank+1)*(height//size) + comm.scan(N,MPI.SUM)

image = np.zeros((ed-st, width, 3))
Y = np.linspace(screen[1], screen[3], height)
X = np.linspace(screen[0], screen[2], width)

for i, y in enumerate(Y[st:ed]):
    for j, x in enumerate(X):
        color = ray_tracing(x,y)
        image[i, j] = np.clip(color, 0, 1)
#	print("%d/%d" % (i + 1, height))

counts = comm.gather(len(image)*width*3, root=0)
recvbuf= None
if rank == 0:
    print(counts)
    recvbuf = np.empty((height,width,3), dtype=float)

# (height, width, 3)
comm.Gatherv(sendbuf=image,recvbuf=(recvbuf,counts), root = 0)
recvbuf = np.array(recvbuf)

# print(recvbuf)
plt.imsave('image3.png', image)

end_time = MPI.Wtime()
if rank == 0: 
    print("Overall elapsed time: " + str(end_time-start_time))
