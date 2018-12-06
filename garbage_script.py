name_sets_for_given_transformation = {
    'exp': get_column_names_from_indices([]),
    'log': get_column_names_from_indices([]),
    'cube': get_column_names_from_indices([]),
    'square': get_column_names_from_indices([]),
    'root': get_column_names_from_indices([])
}


transformation_names = ['exp', 'log', 'cube', 'square', 'root']
exp = lambda x: math.exp(x)
log = lambda x: math.log2(x + 1)
cube = lambda x: x*x*x
square = lambda x: x*x
root = lambda x: math.sqrt(x)

transformations = {
    'exp': exp,
    'log': log,
    'cube': cube,
    'square': square,
    'root': root
}


for transformation_name in transformation_names:
    for column_name in name_sets_for_given_transformation[transformation_name]:
        transformation = transformations[transformation_name]
        trainX[column_name + transformation_name] = trainX[column_name].apply(transformation)
