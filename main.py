#using pdfminer library
import requests
from tqdm import tqdm
import json

user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.101 Safari/537.36'
headers = { 'User-Agent' : user_agent }

# defining keywords to filter articles on corpus
keywords = ["pre frontal cortex", "posterior parietal cortex", "pfc", "prefrontal cortex", "posteriorparietal cortex"]


BASE_URL = "https://www.semanticscholar.org/"
file_index = 1
line_index = 1

# duntion to generate corpus file number in string form
def get_corpus_index(ci):
    if ci/10 < 1:
        return "0"+ str(ci)
    else:
        return str(ci)

# looping over all corpus files 
for i in tqdm(range(40)):
    with open("corpus/s2-corpus-" + get_corpus_index(i)) as file:
        print("ci : ", i)
        # iterating over all articles(lines) in the corpus file
        for line in file:
            found = False
            #print("")
            #print(line_index)
            line_index += 1
            j = json.loads(line)
            #print(j["entities"])

            # searching for keyword in entities
            for term in j["entities"]:
                if term.lower() in keywords:
                    found = True
                    print("found keyword in entities", term, line_index)
                    d_url = j["s2PdfUrl"]
                    if len(d_url) != 0:
                        print("found download")
                        r = requests.get(d_url)
                        with open("extracted/" + str(file_index), 'wb') as f:  
                            f.write(r.content)
                        print("downloaded file")
                        f.close()
                        file_index += 1
                        break
                    else:
                        print("download url not found")
                        break

            # searching for keywords in abstract if not found in entities      
            if (not found):
                abs = j["paperAbstract"]
                abs = abs.lower()
                for word in keywords:
                    if(word in abs):
                        print("found keyword in abstract", word, line_index)
                        # get url from article
                        d_url = j["s2PdfUrl"]
                        if len(d_url) != 0:
                            print("found download")
                            # using requests to download from url
                            r = requests.get(d_url)
                            with open("extracted/" + str(file_index), 'wb') as f:  
                                f.write(r.content)
                            print("downloaded file")
                            f.close()
                            file_index += 1
                            break
                        else:
                            print("download url not found")
                            break


    file.close()