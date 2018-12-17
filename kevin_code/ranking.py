import kevin_utils as util
import operator

name2num = {}


def load_numpy(filename):
    return util.np.load(filename)

def load_order(filename):
    ordering = []
    with open(filename, 'r') as f:
        for line in f:
            ordering.append(line.split('\n')[0])
    return ordering


def cluster_slides(ordering):
    slide_clusters = {}
    cluster_num = 0
    cluster_head = ''
    for x in range(len(ordering)):
        lectureTitle = ordering[x].split('##')[3]
        if lectureTitle != cluster_head:
            cluster_num += 1
            cluster_head = lectureTitle
        slide_clusters[x] = cluster_num
    print 'There are', cluster_num, 'clusters'
    return slide_clusters

def is_adjacent(slide_clusters, slide_num1, slide_num2):
    if slide_clusters[slide_num1] == slide_clusters[slide_num2]:
        return True
    return False

def rank_slide_results(similarities, ordering, slide_clusters, slide_num, distance_weight):
    rankings = []
    #top_score = similarities[slide_num].max() - 0.1 * util.np.ptp(similarities[slide_num])
    #top_similarities = [x for x in util.np.flip(util.np.argsort(similarities[slide_num])) if similarities[slide_num][x] > top_score]
    top_similarities = util.np.flip(util.np.argsort(similarities[slide_num]))[:30]
    print top_similarities
    for x in range(len(top_similarities)):
        if not is_adjacent(slide_clusters, slide_num, top_similarities[x]):
            rankings.append(top_similarities[x])
            break
    while len(rankings) < 10:
        weighted_distance_array = []
        weighted_distance_ordering = []
        all_zero_result = 1
        for x in range(len(top_similarities)):
            if not is_adjacent(slide_clusters, slide_num, top_similarities[x]) and top_similarities[x] not in rankings:
                weighted_distance_array.append(similarities[slide_num][top_similarities[x]])
                all_zero_result = 0
            else:
                weighted_distance_array.append(0)
            weighted_distance_ordering.append(top_similarities[x])
        if all_zero_result:
            return rankings
        for y in range(len(weighted_distance_array)):
            weighted_distance_array[y] *= (1 - distance_weight)
            weighted_distance_array[y] += weighted_distance(rankings, ordering, similarities, top_similarities[y], distance_weight, slide_clusters)
        rankings.append(weighted_distance_ordering[util.np.flip(util.np.argsort(weighted_distance_array))[0]])
    return rankings

def rank_results(similarities, ordering, slide_clusters, filename, distance_weight):
    with open(filename, 'w') as f:
        f.write('')
    with open(filename, 'a') as f:
        for x in range(len(ordering)):
            rankings = rank_slide_results(similarities, ordering, slide_clusters, x, distance_weight)
            f.write(ordering[x])
            for y in rankings:
                f.write(',' + str(ordering[y]) + ',' + str(similarities[x][y]))
            f.write('\n')
            print 'Finished', ordering[x]

def weighted_distance(result_list, ordering, similarities, slide_num, distance_weight, slide_clusters):
    distances = []
    is_not_adjacent = 1
    for slide_num2 in result_list:
        if is_adjacent(slide_clusters, slide_num, slide_num2):
            is_not_adjacent = 0            
        distances.append(1 - similarities[slide_num][slide_num2])
    if is_not_adjacent:
        return distance_weight
    else:
        return distance_weight * util.np.mean(distances)


def main(weight):
    distance_weight = weight
    similarities = load_numpy('./data/Logistic_regression_8_similarity.npy')
    ordering = load_order('./data/slide_names_for_training.txt')
    for x in range(len(ordering)):
        name2num[ordering[x]] = x
    slide_clusters = cluster_slides(ordering)
    rank_results(similarities, ordering, slide_clusters, 'ranking_results_top30_' + str(weight) + '.csv', distance_weight)

for x in range(3):
    main(0.4 + x * 0.1)