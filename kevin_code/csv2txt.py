
def csv2txt(filename, targetfile):
    rankings = []
    with open(filename, 'r') as f:
        for line in f:
            splitLine = line.split(',')
            slide_rankings = [splitLine[0]]
            for x in range(len(splitLine)//2):
                slide_rankings.append(splitLine[x * 2 + 1])
            rankings.append(slide_rankings)
    with open(targetfile, 'w') as f:
        f.write('')
    with open(targetfile, 'a') as f:
        for x in range(len(rankings)):
            f.write(rankings[x][0] + '\nsimilar\n[')
            for result in rankings[x][1:]:
                f.write(result + '\n')
            f.write(']\n')

#csv2txt('ranking_results_0.0.csv', 'ranking_results_0.0.txt')
#csv2txt('ranking_results_0.2.csv', 'ranking_results_0.2.txt')
csv2txt('ranking_results_0.4.csv', 'ranking_results_0.4.txt')
csv2txt('ranking_results_0.5.csv', 'ranking_results_0.5.txt')
csv2txt('ranking_results_0.6.csv', 'ranking_results_0.6.txt')
#csv2txt('ranking_results_0.8.csv', 'ranking_results_0.8.txt')
#csv2txt('ranking_results_1.0.csv', 'ranking_results_1.0.txt')