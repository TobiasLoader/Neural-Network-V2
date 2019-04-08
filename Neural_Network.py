
from math import *
from random import *
from sys import exit


nodesPerLayer = [2, 2]

random_range = [0, 100]
random_range_avg = sum(random_range)/2


def destandardise_range(K):
    return (K*random_range_avg)+random_range_avg


def generate_inputs_list(training_num):
    l = []
    for i in range(training_num):
        valsToAdd = []
        maxInExample = -1
        for j in range(nodesPerLayer[0]):
            val = uniform(-1, 1)
            if val > maxInExample:
                maxInExample = val
                maxPos = j
            valsToAdd.append(val)
        l.append([valsToAdd])
        answers = []
        for j in range(nodesPerLayer[0]):
            if j == maxPos:
                answers.append(1)
            else:
                answers.append(0) # or -1
        l[i].append(answers)
    return l


def init_node_structure():
    x = []
    for layer in range(len(nodesPerLayer)):
        x.append([])
        for node in range(nodesPerLayer[layer]):
            x[layer].append(0)
    return x


def init_weight_structure():
    x = []
    for layer in range(len(nodesPerLayer) - 1):
        x.append([])
        wireNum = nodesPerLayer[layer] * nodesPerLayer[layer + 1]
        for wire in range(wireNum):
            tempW = 0
            if len(nodesPerLayer) > 2:
                while not tempW:
                    tempW = uniform(0, 0.00000000001)
            x[layer].append(tempW)
    return x

def sigmoid(x):
    return 1 / (1 + exp(-x))


def round_n_to_rdp(n, r):
    return round((10**r)*n)/(10**r)

#
# def exagerate_outputs(x):
#     return round_n_to_rdp(sigmoid(round_n_to_rdp(x*10, 1)), 1)


def wrong_input():
    print("\nOh no!\nYour input is not compatible with our code!\nSorry...")
    exit()


def output_is_correct(trials, i):
    test = []
    for j in range(nodesPerLayer[-1]):
        test.append(n[-1][j] == 1 and trials[i][-1][j])
    if True in test:
        return True
    return False


###########################


def val_of_node(x, y, N):
    s = 0
    for l in range(nodesPerLayer[x-1]):
        s += N[x-1][l] * w[x-1][y + l * nodesPerLayer[x]]
    # print(N[0])
    return s


def stimulus_function(i, final_node, aim):
    if final_node[i] > aim[i]:
        return -(aim[i] - final_node[i])**2
    else:
        return (aim[i] - final_node[i])**2


def aim_of_prev_node(x, y):
    s = 0
    for m in range(nodesPerLayer[x+1]):
        if w[x][y + m * nodesPerLayer[x]]:
            s += a[x+1][m] * w[x][y + m * nodesPerLayer[x]]
    return (1/nodesPerLayer[x-1]) * s


def new_weight_calc(x, y, a, n):
    return (a[x+1][y % nodesPerLayer[x+1]]) * (n[x][y // nodesPerLayer[x+1]])


def refine_weights():
    for k in range(len(init_weight_structure())):
        m = 0
        for l in w[k]:
            m += abs(l)
        m /= len(w[k])
        for l in range(len(w[k])):
            if m:
                w[k][l] *= (1/m)


def training():
    global training_num
    try:
        training_num = int(input("\nNumber of training examples: "))
    except:
        wrong_input()
    if training_num > 10:
        inner_repeat_num = 10
    else:
        inner_repeat_num = round(sqrt(training_num))
    # inner_repeat_num = 1
    inputs = generate_inputs_list(training_num)
    # inputs = [[[0.25159101560631214, 0.4065541379070743, 0.04999604933458368], [0, 1, 0]], [[0.16745798393340516, 0.10012964854145023, -0.13982259637706185], [1, 0, 0]], [[-0.1032428919114845, 0.6563412505730761, -0.4326332135287243], [0, 1, 0]], [[-0.7538384298432057, 0.538167120827955, 0.19272589842544985], [0, 1, 0]], [[0.1368845673837642, 0.649749734610799, 0.2961162951647749], [0, 1, 0]], [[0.14187718859113319, 0.34144568449510704, -0.33245212810395364], [0, 1, 0]], [[0.9209710915701101, -0.5071817427302594, 0.5466043856562914], [1, 0, 0]], [[-0.03578478761232562, 0.8589059248746371, -0.30263604918092857], [0, 1, 0]], [[-0.44544375626097144, 0.5800378732403848, -0.42432939206732323], [0, 1, 0]], [[0.8854545430375829, 0.18774655640496807, 0.015200057379719079], [1, 0, 0]], [[-0.42235770697373853, 0.48246791832015945, -0.4095546050766532], [0, 1, 0]], [[-0.8252451705192885, -0.4692092551202174, -0.6393538659549862], [0, 1, 0]], [[-0.6608971865937501, -0.932302140967145, 0.7211835879042781], [0, 0, 1]], [[0.3052659688651529, 0.7686159840673441, -0.6997464288415849], [0, 1, 0]], [[0.8310299064167141, -0.741373928433938, 0.14663247640621768], [1, 0, 0]], [[0.3862570810626016, 0.9716358575025377, 0.4630384613297014], [0, 1, 0]], [[0.3030673549561831, 0.3580768824469618, -0.288727252335115], [0, 1, 0]], [[-0.36976369020163147, 0.46504913700374906, 0.8275636191536171], [0, 0, 1]], [[0.19162741905540392, -0.6568917385059219, -0.7459735584609328], [1, 0, 0]], [[-0.014534606928358063, 0.19365029448212767, -0.8652675951281428], [0, 1, 0]]]
    new_w = init_weight_structure()
    for i in range(len(inputs)):
        n = init_node_structure()
        n[0] = inputs[i][0]
        # n[0] = inputs[i][0]
        # print(n, end='')
        final_node_temp = init_node_structure()[-1]

        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-1):
                if j:
                    for k in range(nodesPerLayer[j]):
                        n[j][k] = val_of_node(j, k, n)

        for j in range(nodesPerLayer[-1]):
            final_node_temp[j] = val_of_node(len(nodesPerLayer)-1, j, n)

        for j in range(nodesPerLayer[-1]):
            a[-1][j] = stimulus_function(j, final_node_temp, inputs[i][1])

        n[-1] = final_node_temp

        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-2):
                k = len(nodesPerLayer) - j - 2
                for l in range(nodesPerLayer[k]):
                    a[k][l] = aim_of_prev_node(k, l)

        for j in range(len(init_weight_structure())):
            for k in range(len(init_weight_structure()[j])):
                new_w[j][k] += new_weight_calc(j, k, a, n)

        # print(n)

        if i and not i % inner_repeat_num:
            for j in range(len(init_weight_structure())):
                avg = 0
                for k in range(len(w[j])):
                    w[j][k] += sigmoid(new_w[j][k] / inner_repeat_num)
                    avg += w[j][k]
                avg /= len(init_weight_structure()[j])
                for k in range(len(w[j])):
                    w[j][k] -= avg
                refine_weights()
            new_w = init_weight_structure()

    refine_weights()


def trialling():
    global trial_num
    try:
        trial_num = int(input("Number of actual trials after training: "))
    except:
        wrong_input()
    try:
        show_incorrect = int(input("Show the incorrect determinations (0 = no, 1 = yes): "))
    except:
        wrong_input()
    correct_num = 0
    trials = generate_inputs_list(trial_num)
    if show_incorrect:
        print("\nINCORRECT:\n")
    for i in range(trial_num):
        n[0] = trials[i][0]

        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-1):
                if j:
                    for k in range(nodesPerLayer[j]):
                        n[j][k] = val_of_node(j, k, n)
        maxNeg = 0
        for j in range(nodesPerLayer[-1]):
            n[-1][j] = (val_of_node(len(nodesPerLayer)-1, j, n))
            if n[-1][j] < 0 and abs(n[-1][j]) > maxNeg:
                maxNeg = abs(n[-1][j])

        maxVal = 0
        for j in range(nodesPerLayer[-1]):
            n[-1][j] += maxNeg
            if n[-1][j] > maxVal:
                maxVal = n[-1][j]

        for j in range(nodesPerLayer[-1]):
            n[-1][j] /= maxVal

        if output_is_correct(trials, i):
            correct_num += 1
        elif show_incorrect:
            for j in range(nodesPerLayer[0]):
                n[0][j] = destandardise_range(n[0][j])
            print(" -", n)
    print('\nThe neural network was correct', 100 * correct_num / trial_num, '% of the time.\n')


play = True
while play:
    inputs = []
    n = init_node_structure()
    w = init_weight_structure()
    a = init_node_structure()
    training()
    trialling()
    for j in range(nodesPerLayer[0]):
        n[0][j] = destandardise_range(n[0][j])
    print("Last example:", n)
    print("Final weights:", w, "\n")
    if input("Again [Y/n]: ") not in ["Y", "y"]:
        play = False
    print("")

exit()