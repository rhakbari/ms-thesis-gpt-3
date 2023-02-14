import matplotlib.pyplot as plt

# Data
steps = [i for i in range(1, 18)]
training_loss = [0.113503868134496, 0.0745849757946455, 0.0532221682766566, 0.0886128272311163, 0.0294547074381965, 0.0666500272092219, 0.018294738624273, 0.0326645254992553, 0.00635636508243152, 0.0194682271687814, 0.00973232916588536, 0.0213782167385014, 0.00810845594830153, 0.010824551640359, 0.00641220797419131, 0.020386791810267, 0.0198714041643729]
validation_loss = [0.493259494011258, None, None, None, None, None, None, None, 0.0871569393571882, None, None, None, None, None, None, None, 0.340534550387619]
training_sequence_accuracy = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
validation_sequence_accuracy = [0, None, None, None, None, None, None, None, 0, None, None, None, None, None, None, None, 0]

# Create figure and axis
fig, ax1 = plt.subplots()

# Plot training and validation loss
ax1.plot(steps, training_loss, 'b-', label='Training Loss')
ax1.plot(steps, validation_loss, 'r-', label='Validation Loss')

# Set y-axis label
ax1.set_ylabel('Loss')

# Create second y-axis for sequence accuracy
ax2 = ax1.twinx()

# Plot training and validation sequence accuracy
ax2.plot(steps, training_sequence_accuracy, 'g-', label='Training Sequence Accuracy')
ax2.plot(steps, validation_sequence_accuracy, 'y-', label='Validation Sequence Accuracy')

# Set y-axis label
ax2.set_ylabel('Sequence Accuracy')

# Add legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Set x-axis label
plt.xlabel('Steps')

# Show plot
plt.show()
