import numpy as np

def generate_multiclass_data(num_samples, class_parameters):
    data = []
    labels = []
    
    for class_label, (mean, std) in enumerate(class_parameters):
        class_data = np.random.normal(loc=mean, scale=std, size=(num_samples, 2))
        class_labels = np.full((num_samples, 1), class_label)
        
        data.append(class_data)
        labels.append(class_labels)
    
    # Concatenate data and labels for each class
    data = np.vstack(data)
    labels = np.vstack(labels)
    
    return data, labels.flatten()