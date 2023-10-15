import math
class Node:
    def __init__(self):
        self.value = None
        self.children = {}

def entropy(data):
    """Calculate entropy for a given set of data."""
    total_records = len(data)
    if total_records == 0:
        return 0
    positive_count = sum(1 for record in data if record[-1] == 'Yes')
    negative_count = total_records - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0
    positive_prob = positive_count / total_records
    negative_prob = negative_count / total_records
    return -positive_prob * math.log2(positive_prob) - negative_prob * math.log2(negative_prob)

def information_gain(data, attribute_index):
    """Calculate information gain for a given attribute."""
    total_entropy = entropy(data)
    attribute_values = set(record[attribute_index] for record in data)
    weighted_entropy = 0
    for value in attribute_values:
        subset = [record for record in data if record[attribute_index] == value]
        subset_entropy = entropy(subset)
        subset_weight = len(subset) / len(data)
        weighted_entropy += subset_weight * subset_entropy
    return total_entropy - weighted_entropy

def choose_best_attribute(data, attributes):
    """Choose the best attribute based on information gain."""
    gains = [information_gain(data, i) for i in range(len(attributes) - 1)]
    max_gain = max(gains)
    best_attribute_index = gains.index(max_gain)
    return best_attribute_index

def build_tree(data, attributes):
    """Build the decision tree using the ID3 algorithm."""
    classes = [record[-1] for record in data]
    
    # If all examples belong to the same class, return a leaf node
    if len(set(classes)) == 1:
        leaf_node = Node()
        leaf_node.value = classes[0]
        return leaf_node
    
    # If there are no more attributes to split on, return a leaf node with the majority class
    if len(attributes) == 1:
        majority_class = max(set(classes), key=classes.count)
        leaf_node = Node()
        leaf_node.value = majority_class
        return leaf_node

    best_attribute_index = choose_best_attribute(data, attributes)
    best_attribute = attributes[best_attribute_index]
    tree = Node()
    tree.value = best_attribute
    
    # Remove the best attribute from the list of attributes
    attributes = [attr for attr in attributes if attr != best_attribute]
    attribute_values = set(record[best_attribute_index] for record in data)
    for value in attribute_values:
        subset = [record for record in data if record[best_attribute_index] == value]
        child_node = build_tree(subset, attributes)
        tree.children[value] = child_node
    return tree

def print_tree(node, indent=''):
    """Print the decision tree."""
    if not node.children:
        print(indent + 'Class:', node.value)
        return
    print(indent + 'Attribute:', node.value)
    for value, child_node in node.children.items():
        print(indent + ' Value', value)
        print_tree(child_node, indent + ' ')

# Sample dataset (Weather, Temperature, Humidity, Wind, Play)
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Attribute names (excluding the target attribute 'Play')
attributes = ['Weather', 'Temperature', 'Humidity', 'Wind']

# Build the decision tree
root_node = build_tree(data, attributes)
print_tree(root_node)
