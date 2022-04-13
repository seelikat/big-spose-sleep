import csv

if __name__=="__main__":

    phrases = []
    
    with open("decoded_answers.csv", newline='') as phrasescsv:
        phrasescsvreader = csv.DictReader(phrasescsv, delimiter=',')
        for row in phrasescsvreader:
            phrases.append( row['decoded_feature'] )
    
    with open("gpt3semantics.txt", 'w') as handle:
        for phrase in phrases: 
            handle.write('%s\n' % phrase)
