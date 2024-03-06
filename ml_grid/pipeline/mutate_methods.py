


import random




def baseLearnerGenerator(ml_grid_object):
    
    modelFuncList = ml_grid_object.modelFuncList
    
    index = random.randint(0, len(modelFuncList)-1)

    return modelFuncList[index]() #store as functions, pass as result of executed function


# Model will be fit in generation stage and pass fitted state with training data.


def mutateEnsemble(individual, ml_grid_object):
    # print(individual[0])
    # print(len(individual[0]))
    try:
        print(f"original individual of size {len(individual[0])-1}:")
        n = random.randint(0, len(individual[0])-1)
        print(f"Mutating individual at index {n}")
        try:
            individual[0].pop(n)
            print(f"Successfully popped {n} from individual")
        except Exception as e:
            print(f"Failed to pop {n} from individual of length {len(individual[0])} , popping zero")
            individual[0].pop(0)
            
            print(e)
        
        
        
        # print(n)
        #individual[0].pop(n)
        # print(individual)
        individual[0].append(baseLearnerGenerator(ml_grid_object))
        # print(individual)
        #print("Mutated individual:")
        #print(individual)
        return individual
    except Exception as e:
        print(e)
        print("Failed to mutate Ensemble")
        print("Len individual", len(individual))
        #print(individual[0])
        #print(individual)
        return individual