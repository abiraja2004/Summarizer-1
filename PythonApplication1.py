from nltk.tokenize import word_tokenize as wt, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag_sents
from nltk.tag import pos_tag
from nltk.tag import map_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.classify.megam import numpy
from string import punctuation
import numpy
from array import array
from nltk.corpus import wordnet as wn,brown
from array import array
from numpy import matrix
import networkx as nx
import matplotlib.pyplot as plt
import math

import tkinter
from tkinter import ttk

class Adder(ttk.Frame):
    #sentence = " RAM keeps things being worked with. The CPU uses RAM as a short-term memory store. Big RAM size and CPU helps speed. Without them computer do not work"
    sentence =""
    sentences = []
    temp=[]
    nouns=""
    f1=[]
    f2=[]
    col=[]
    graph=[]
    count=0

    def word_sim(self,w1,w2) :
        flag=0
        flag1=0
        word1= ""
        word2 = ""
   
        print(w2)
        word1 =pos_tag([w1])
        word2 =pos_tag([w2])
        print(w1+' '+word1[0][1])
        print(w2+' '+word2[0][1])
        if word1[0][1].startswith('N') :
      
         w1  =  w1+".n.01"
         flag=1
        elif word1[0][1].startswith('JJ') :
         w1  =  w1+".a.01"
         flag=1
        elif word1[0][1].startswith('RB') :
         w1  =  w1+".r.01"
         flag=1
        if word2[0][1].startswith('N') :
         w2  =  w2+".n.01"
         flag1=1
        elif word2[0][1].startswith('JJ') :
         w2  =  w2+".a.01"
         flag1=1
        elif word2[0][1].startswith('RB') :
         w2  =  w2+".r.01"
         flag1=1
        if(flag==1 and flag1==1) :
          try :
            w11=wn.synset(w1)
            w21=wn.synset(w2)
          except :
            print("end2") 
            return 0.0;
        else :
            print("end1") 
            return 0.0;       
        print("end "+w1+" "+w2)       
  
        if w11.path_similarity(w21) is not None :
           return w11.path_similarity(w21);   
      
        return 0.0;  
    def cek_corpus(self,word) :
        #genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
        genres = ['science_fiction']
        count=0;
    
        for genre in genres :
            news_text = brown.words(categories=genre)
            w=0
            while(w<len(news_text)):
                if news_text[w]==word :
                 count = count+1             
                w=w+1
        return count;
    def calculate_corpus(self,s1,f1,f2) :
    
        i=0
        N=14470
        while i<len(f1) :
            if s1[i]!=0 :
             n1=self.cek_corpus(f1[i])
             f11=math.log(n1+1)/math.log(N+1)
             f11=f11-1
             n2=self.cek_corpus(f2[i])+1
             f12=math.log(n2+1)/math.log(N+1)
             f12=f12-1
             print("f",s1[i]*f11*f12)
             s1[i]=s1[i]*f11*f12
            i=i+1
        return s1;
    def table_cells (self,sent3,sent1,sent2) :  
        #matrix = [[0 for x in range(len(sent1)+len(sent1))] for y in range(len(sent1))]
        i=0
        global f1,f2,col
        f1=[]
        f2=[]
        col=[]
   
        matrix=numpy.zeros((len(sent1), len(sent3)))
        s=numpy.zeros ((len(sent3)))
        col=numpy.zeros((len(sent3)))
        f1=["" for x in range(len(sent3))]
        f2=["" for x in range(len(sent3))]
        print(matrix)
        print(s)
        """ 
        for word in sent1 :
            for word1 in sent2 :   
               if word1 != word   : 
                    print(word1)
                    print(word)
                    temp+=word1+" " 
         """            
        print(sent3)
        i=0
        j=0
        while i<len(sent1) :
            j=0
            print("word[i]"+sent1[i])
            while j<len(sent3) :     
             if sent1[i]==sent3[j] :
                matrix[i][j]=1
             else :
               #print(word_sim(sent1[i],sent3[j]))
               if self.word_sim(sent1[i],sent3[j]) >=0.2 :
                 matrix[i][j]=self.word_sim(sent1[i],sent3[j])
           
             if matrix[i][j]>s[j] :
                s[j]=matrix[i][j] 
                f1[j] = sent1[i]
                f2[j] = sent3[j]
                col[j] = i+1
             if i==len(sent1)-1 and s[j]==0 :  
                 f1[j] = sent3[j]
                 f2[j] = sent3[j]
             j=j+1
            i=i+1
    
    
        print(s)    
        print(matrix)
        return s;

    def calculate_row(self,s) :
          i=0
          r=numpy.zeros(len(s))
          print("f1 in row",f1)
          print("f1 in row",f2)  
          print("col",col)
          while i<len(s) :

    
               if f1[i]==f2[i] and s[i]!=0 :
                 r[i]=col[i]
               else :
                if s[i]>=0.2 :
                  r[i]=col[i] 
                else :
                  r[i]=0   
               i=i+1      
          return r ;
    """
    def cosine_similarity(vector1,vector2):
      float(dot)
      # Calculate numerator of cosine similarity
      dot = [vector1[i] * vector2[i] for i in len(vector1)]
  
      # Normalize the first vector
      sum_vector1 = 0.0
      sum_vector1 += sum_vector1 + (vector1[i]*vector1[i] for i in len(vector1))
      norm_vector1 = sqrt(sum_vector1)
  
      # Normalize the second vector
      sum_vector2 = 0.0
      sum_vector2 += sum_vector2 + (vector2[i]*vector2[i] for i in len(vector2))
      norm_vector2 = sqrt(sum_vector2)
  
      return float(dot/(norm_vector1*norm_vector2))
    """
    def scalar(self,collection):  
      total = 0.0 
      i=0
      while i<len(collection):  
        total += float(collection[i]*collection[i])  
        i+=1
      return float(math.sqrt(total))  
    
 
    def similarity(self,A,B): # A and B are coin collections  
       total = 0.0  
       kind=0
       #for kind in A: # kind of coin  
         #if kind in B: 
       while kind<len(A) :
           total += A[kind] * B[kind]  
           kind+=1
       return float(total) / (self.scalar(A) * self.scalar(B)) 

    def cosine_similarity(self,vector1, vector2):
        dot_product = sum(p*q for p,q in zip(vector1, vector2))
        magnitude = math.sqrt(float(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2])))
        if not magnitude:
            return 0
        return float(0.85*dot_product)/magnitude

    def calculate_sentence(self,s1,s2,r1,r2) :
        ss =self.similarity(s1,s2)
        print(ss)
        print()
        sr=(0.15)*numpy.linalg.norm(numpy.subtract(r1,r2))
        sr1= numpy.linalg.norm(numpy.add(r1,r2))
        print(sr)
        print(sr1)
        if sr1!=0 :
            sr=sr/sr1
        stt= (ss)+(sr)
        print(stt)
        return stt ;

    def sent_conc(self,sent1,sent2) :
        sent3 = []
        sent3 = sent1+sent2
        i=0
        while i<len(sent1) :
            j=len(sent1)
            while j<len(sent3) :
                if sent1[i]==sent3[j] :
                    sent3.pop(j)   
                j=j+1
            i=i+1 
        return sent3;
    
    def prepros_sent(self,sent1 , sent2,a,b) :
        global graph,count
        print("pre")
        #sent1 = [token for token in wt(sent1) if  token not in punctuation]
        #sent2 = [token for token in wt(sent2) if token not in punctuation] 
        sent3=self.sent_conc(sent1,sent2)
        s1=self.table_cells (sent3,sent1, sent2)
        s1=self.calculate_corpus(s1,f1,f2);
        r1 = self.calculate_row(s1)
        print(s1)
        print(r1)

        s2=self.table_cells (sent3,sent2, sent1)
        s2=self.calculate_corpus(s2,f1,f2);
        r2 = self.calculate_row(s2)
        print(s2)
        print(r2)
        stt=self.calculate_sentence(s1,s2,r1,r2) 
        if stt>=0.5 :
            graph[a][b]=1
            self.count =self.count+1
        return ;

    def summarize(self) :
        G = nx.DiGraph(graph)
        pos1 = nx.shell_layout(G)
        #pos1 = [[0,0], [0,1], [1,0]]
        print(nx.pagerank_numpy( G,0.85))
        scores=nx.pagerank_numpy( G,0.85)
        ranked=[]
        ranked=sorted(((scores[i],i) for i,s in enumerate(sentences) ),reverse=True)
        i=0
        j=0
        final=[]
        while (i<len(sentences)/2) :
             final.append(ranked[i][1])
             i+=1
        final=sorted(final)
   
        while(j<len(final)) :
            self.answer_label['text'] += sentences[final[j]] +"\n" 
            print(sentences[final[j]])
            j+=1    
        print(graph)               
        nx.draw(G,pos1,with_labels=True)
        plt.show()
        return;

    def tok_lem(self) :
       lss = LancasterStemmer()
       wnl = WordNetLemmatizer()
       i=0
       while i <len(temp)  :      
            temp[i] = [token for token, pos in pos_tag(wt(temp[i])) if pos.startswith('N') or pos.startswith('JJ') or pos.startswith('RB') and token not in punctuation]       
            i= i+1
    
       i=0
   
       while i<len(temp) : 
         j=0
         while j<len(temp[i]) :
          temp[i][j] = wnl.lemmatize(temp[i][j])
          j=j+1
         i=i+1
       return;

    def compare_sent(self):
        print(self.num1_entry.get("1.0",'end-1c'))
        global sentence,sentences,temp,graph
        sentence=self.num1_entry.get("1.0",'end-1c')
        sentences=sent_tokenize(sentence)
        temp=sent_tokenize(sentence)
        print(temp)
        graph=numpy.zeros((len(sentences), len(sentences)))
        self.tok_lem()
        i=0
        while i<len(temp) :
            j=i+1
            while j<len(temp) :
                self.prepros_sent(temp[i],temp[j],i,j)
                j=j+1
            i=i+1
        self.summarize()
   
        return ;


    #The adders gui and functions.
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def on_quit(self):
       #Exits program.
        quit()

    #def calculate(self):
       #Calculates the sum of the two inputted numbers.
        #num1 = int(self.num1_entry.get())
        #num2 = int(self.num2_entry.get())
        #num3 = num1 + num2
        #sentences = sent_tokenize("Our system, based on LexRank ranked in first place in more than one task in the recent DUC 2004 evaluation. In this paper we present a detailed analysis of our approach and apply it to a larger data set including data from earlier DUC evaluations. We discuss several methods to compute centrality using the similarity graph. The results show that degree-based methods (including LexRank) outperform both centroid-based methods and other systems participating in DUC in most of the cases. Furthermore, the LexRank with threshold method outperforms the other degree-based techniques including continuous LexRank. We also show that our approach is quite insensitive to the noise in the data that may result from an imperfect topical clustering of documents.")
       

    def init_gui(self):
       
        #Builds GUI.
        self.root.title('Summarizer')
        self.root.option_add('*tearOff', 'FALSE')

        self.grid(column=0, row=0, sticky='nsew')

        self.menubar = tkinter.Menu(self.root)

        self.menu_file = tkinter.Menu(self.menubar)
        self.menu_file.add_command(label='Exit', command=self.on_quit)

        self.menu_edit = tkinter.Menu(self.menubar)

        #self.menubar.add_cascade(menu=self.menu_file, label='File')
        #self.menubar.add_cascade(menu=self.menu_edit, label='Edit')

        #self.root.config(menu=self.menubar)

        self.num1_entry = ttk.tkinter.Text(self, height=5, width=100)
        self.num1_entry.grid(column=5, row = 2,ipady=100)
      

        #self.num2_entry = ttk.Entry(self, width=100)
        #self.num2_entry.grid(column=3, row=3)

        self.calc_button = ttk.Button(self, text='Summarized',
                command=self.compare_sent)
        self.calc_button.grid(column=2, row=7, columnspan=4)

        self.answer_frame = ttk.LabelFrame(self, text='Result',width=24)
        self.answer_frame.grid(column=5, row=8, columnspan=150,ipady=100,ipadx=150 ,sticky='nesw')

        self.answer_label = ttk.Label(self.answer_frame, text='')
        self.answer_label.grid(column=5, row=5)

        # Labels that remain constant throughout execution.
        ttk.Label(self, text='Text Summarizer').grid(column=0, row=0,
                columnspan=4)
        ttk.Label(self, text='Text Input :').grid(column=0, row=2,
                sticky='w')
        #ttk.Label(self, text='Result :').grid(column=2, row=2,sticky='w')

        ttk.Separator(self, orient='horizontal').grid(column=0,
                row=1, columnspan=4, sticky='ew')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)

if __name__ == '__main__':
    root = tkinter.Tk()
    Adder(root)
    root.mainloop()

#sentence = "We introduce a stochastic graph-based method for computing relative importance of textual units for Natural Language Processing. We test the technique on the problem of Text Summarization (TS). Extractive TS relies on the concept of sentence salience to identify the most important sentences in a document or set of documents. Salience is typically defined in terms of the presence of particular important words or in terms of similarity to a centroid pseudo-sentence. We consider a new approach, LexRank, for computing sentence importance based on the concept of eigenvector centrality in a graph representation of sentences. In this model, a connectivity matrix based on intra-sentence cosine similarity is used as the adjacency matrix of the graph representation of sentences."
#sentence=" Our system, based on LexRank ranked in first place in more than one task in the recent DUC 2004 evaluation. In this paper we present a detailed analysis of our approach and apply it to a larger data set including data from earlier DUC evaluations. We discuss several methods to compute centrality using the similarity graph. The results show that degree-based methods (including LexRank) outperform both centroid-based methods and other systems participating in DUC in most of the cases. Furthermore, the LexRank with threshold method outperforms the other degree-based techniques including continuous LexRank. We also show that our approach is quite insensitive to the noise in the data that may result from an imperfect topical clustering of documents."
#sentence ="We introduce a stochastic graph-based method for computing relative importance of textual units for Natural Language Processing. We test the technique on the problem of Text Summarization (TS)."
#sentence = "Rina is happy. i barn chicken. if Rina happy she eat cake. i ate bread"




"""
if r[0]!=0 :
    print("a")
 #nltk.ConditionalFreqDist((genre, word)for genre in brown.categories()for word in brown.words(categories=genre))


print(ranked) 
nx.draw(G,pos1,with_labels=True)

#print(pagerank(s, 0.85, 1.0e-8))
#s1=numpy.ones((3))
#print(numpy.add(numpy.divide(numpy.multiply(s1,s),s1),s1))


genres = ['science_fiction']
total=0

for genre in genres :
        news_text = brown.words(categories=genre)
        total=total+len(news_text)
        print()
        
print(total)

#for word,tag in pos_tag(word) :
#    print("%-10s %-10s"%(word, map_tag("en-ptb","universal",tag) ))
i=0
while i <len(sentences)  :
    
    sentences[i] = [token for token, pos in pos_tag(wt(sentences[i])) if pos.startswith('N') or pos.startswith('JJ')]
    print(sentences[i])
    i= i+1
"""

 
