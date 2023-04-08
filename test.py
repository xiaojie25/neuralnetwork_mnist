from python import *

test_data_file = open("mnist_test.csv", 'r')                       
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    # print(correct_label, "correct label")

    input = (np.asfarray(all_values[1:]) /255 * 0.99) + 0.01
    output = n.query(input)

    label = np.argmax(output)
    # print(label, "network's label") 

    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asfarray(scorecard)
performance = scorecard_array.sum() / scorecard_array.size
print(performance)
