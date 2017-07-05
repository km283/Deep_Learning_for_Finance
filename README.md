# Deep_Learning_for_Finance
MOST COMMON NEURAL NETWORK BUILDING BLOCKS DESCRIBED BELOW 
- Image summary available at: http://nyc2016.fpq.io/fpq_yam_peleg_deep_learning.pdf   page12

AIM AND BASICS DESCRIPTION:

Rather than try every neural architecture imaginable, we shall try and mimic best practices applied by investors daily. Hence, in the same way that convolutional neural networks (CNNs) mimic our own visual cortexes, our neural architecture shall mimic the discovery process followed by investors. Since we are focusing on technicals, as well as fundamentals appearing in textual form, we must attempt to find an architecture that looks for price chart patterns similarly as a day traded, and fundamentals as both an arbitrager and a longer term investor. Finally we must identify a method to combine these in an efficient way (see - https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf - for inspiration). 

Taking these in step, day traders (not considering event arbitragers yet) attempt to gauge the direction of the heard and profit from it. They are in affect comparable to cattle herders. In trying to fulfill there aim they primarily concentrate on identifying patters in price charts, looking for PATTERNS appearing in conjunction with the passage of TIME. Elaborating, a day trader may look for support and resistance lines (bands in which prices move), or heads and shoulders (a famous technical patter that some investors swear by), and make a trade based on his belief of the likelihood of this pattern continuing tomorrow, in the next hour or etc. The difficulty in mimicking this kind of analysis is that neither CNNs - which look for patterns and make decisions irrespective of where (time t or t-10) they occur -  nor recurrent neural networks - which try and analyse the cumulative effects of the passage of time, or addition of variables - on their own replicate this process completely. Ideally we would like an architecture that allows us to capture both the presence of patterns and the passage of time. Consequently, we choose to input price and volume data into two parts of our network - one CNN network, in conjunction with multiday 'headlines vectors', and one long short term memory RNN (LSTM). (see architecture section for more details)

*Other option: Consequently, we choose to implement a modern max pooling RNN (quote Michael's paper) when considering price and volume data (see architecture section for more details).*

Moving to another form of short term trading, event arbitragers traditionally focus on reacting to simple (requiring little data to understand) events that create momentary arbitrage opportunities (FOOTNOTE: our use of the word arbitrage here is not entirely accurate, as events do not always move prices in the same way; in the kind of event analysis we are considering some risk is present). An example of this may be someone shorting a stock after it announces its intent to acquire another company; on average companies tend to overpay. Whilst, due to the industry wide automation of event trading, and simple analysis required to spot such opportunities, the trading signal strength of these strategies is short (an optimistic estimate would be one to two days max), we nevertheless add tools to our architecture potentially capture such signals. We follow the approach set out by Ding et al. (2014, 2015) - see summary papers 1 and 2 - utilising gaited recurrent nets (GRNs) (FOOTNOTE: form of advanced RNN. Nikhil Buduma (Deep Learning 2017) notes that GRNs perform equally to LSTMs; however, they require fewer parameters to do so). 

Finally, considering fundamental investors, traditionally these focus on identifying the long term fair price using non trivial historic data. Whilst prices may fluctuate in the short term due to excess buying or selling, in the long run they should, the theory goes, encapsulate the securities fundamental value. Taking an everyday example, whilst the price of sneakers may fluctuate due to seasonal or fashion linked reasons in the short term, in the long therm their price is linked to the companies manufacturing and marketing costs. Whilst as part of this dissertation we do not attempt to calculate a stock's fundamental value ourselves, due to the complexity of performing such analysis, we nevertheless try to capture the thinking involved in fundamental analysis, by analysing the interactions between everyday headlines. Specifically, we try and capture long term company strategic trends, thus improving the accuracy of our event based analysis. As such a task involves looking for patterns in data, we use CNNs for the task. 


Overall, to summarise, we attempt to determine the (specify time frame here)  short and medium direction - buy, sell hold (within threshold) - of stock price movements. Due to the complexities of this task we try and capture the trading signals arising from multiple analysis methods. As these signals are not mutually exclusive, we use a multistage neural network architecture involving RNNs, CNNs, and fully connectected neural networks, to efficiently capture the inputs interactions, thus maximising our prediction accuracy over both time frames. 

ARCHITECTURE & TRAINING: 

With roughly 200k headlines providing roughly 40k training instances we have to use the training data efficiently to accomplish our aim. In this endeavour we have developed a  a modular architecture/approach that allows the reuse of trained weights, and furthermore the use of the unsupervised learning encoder decoder approach (FOOTNOTE: Whilst advances in deep learning have allowed researchers to use techniques to perform end-to-end training, we follow the older method of training our network in steps to provide numerous base case performance figures). We start by describing our handling of an individual headline, before moving on to the analysis of multiple headlines in one day, and then multiple headlines in a span of 20 days. After thus describing our approaches to textual information we describe our handling of price and volume data, before concluding with a description of our overall architecture, which efficiently combines all inputs to reach a trading signal. Note, that whilst we use our training data repeatedly, we strictly avoid using test data at any point in our system's training. 

SIDE NOTE: Word2Vec:
To succeed in analysing textual information using neural networks, the information must first be transformed into a numeric format. Specifically, using an approach termed word2vec, we can convert a headline of strings into a sequence of vectors. Elaborating, every word can be represented as a point in an n dimensional space. This point conveys the context in which the word is used, a characteristic visible by examining the words appearing near it in the n dimensional space. Demonstrating through an example, the words cat and dog will form points closer together than the words cat and France. (FOOTNOTE: whilst cat and dog are two different animals, we could for instance substitute them both into the sentence 'i have a four legged X in my house' . Whilst the sentence itself would thus convey something different, we cannot for instance substitute the word France into it, characterising the contextual similarity between cat and dog, but not between cat and France.) We learn these representation by passing one hot encoded vectors through an encoder and storing the network's weights (skip-gram model). We use softmax negative sampling (NCE as outlined in Deep Learning 2017 page 146) to train these weights. (for more information see: http://web2.cs.columbia.edu/~blei/seminar/2016_discrete_data/readings/MikolovSutskeverChenCorradoDean2013.pdf). 

IMPLEMENTATION LINKS FOR ABOVE: 
Deep Learning page 146
GLOVE implementation: 
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class2/glove.py

*add more about our approach - do we use the 8mil headline corpus to learn the vectors, or do we reuse Richard's weights. Finally note the vector dimension chosen. Whilst Richard's paper points to using 300 dimensional vectors, we could quote Ding et al. who used 100 dimensions.*

One Headline: 
Our aim is to capture as much of the headline's/sentence's meaning and represent this as a point. Were we solely interested in performing arbitrage event investing, we could subsequently - using these headline point representations - look for clusters of points leading to either buy, hold, or sell decisions. Consequently we could than predict if a new test case headline is a buy or etc depending on its position in the n dimensional space. 

To train a network to summarise a sentence as a point, we could use a similar procedure to word2vec called doc2vec (see https://cs.stanford.edu/~quocle/paragraph_vector.pdf), which transforms sequences of text to paragraph vectors. Alternatively, we could use a summarisation tool outlined in Ding et al. (2014), which first requires preprocessing of headlines into OPO format (summarised in previous document) before using a recursive neural network (RNTN) to calculate the equivalent of a paragraph vector. Finally, we could use GRN encoder decoder, feeding in the sequence of text and storing the final output (midpoint of seq2seq). We choose to implement the final method, describing our training approach below. 

In training our headline encoder decoder we could either use a loss function linked to: company returns, or to replicating the original input sequence (input_seq2input_seq). As multiple headlines can appear in any given day, we choose the later of these (FOOTNOTE: this should further lead to a better network, as the task is closer linked to our aim of obtaining a semantic representation of the headline). Hence, we train the encoder decoder by passing in d - d is the maximum identified headline length in the training & test sample, with zero padding implemented to make all headlines of equal size thereafter - 100/300 (link to choice made italic part of document) dimensional word vectors, and comparing the outputs with the original inputs. We use a seq2seq cross entropy loss function (outlined in: https://www.tensorflow.org/tutorials/seq2seq), or the cosine similarity loss function (see Deep Learning page 220 I believe) to train the encoder weights. Overall the encoder learns to represent the word vectors in 100/300 dimensions, which due to the relatively small size of our training set, we reduce using pca whitening to 30 dimensions in our complete network (outlined below).  

*Note above I mention using PCA whitening. After reading Deep Learning I am led to believe that it would be better to have the encoder perform the dimensionality reduction. As Nikhil Buduma argues, PCA strugles to capture non linear relationships, meaning that the use of a NN encoder tends to better capture the important dimensions (see page 120 of Deep Learning)

**Note due to the sparsity of training data we only consider a network with a maximum of two hidden layers.* 

***Note, Deep Learning outlines the use of cosine distance as a loss function when comparing seq2seq. Further, Deep Learning provides a more advanced framework for padding (split sentences into buckets of certain lengths to reduce the overall quantity of padding required - this should tackle any vanishing gradient problem and speed up training). We may not have the necessary data for the second step which is why we implemented standard padding.

IMPLEMENTATION LINKS: 
Deep Learning  pages 184 - end of chapter
https://github.com/ematvey/tensorflow-seq2seq-tutorials


Multiple Headlines Appearing in One Day: 
We outlined in the introduction that our aim when analyzing multiple headlines is to capture complex strategic trends. Whilst we proposed using CNNs to achieve this, we prefer to use a max pooling RNN whilst analysing multiple headlines appearing in ONE DAY (great suggestion Michael). Elaborating, we believe that capturing the overall importance of headlines appearing in one day will be more useful when analysing patterns appearing between days. We avoid taking averages of headlines as Ding et al. per the suggestions of Mikolov 2014 (paper already quoted :https://cs.stanford.edu/~quocle/paragraph_vector.pdf). Further, our approach allows us to avoid discarding stories not classified as 'top stories'. 

Describing the training approach in detail, as with an individual sentence we measure the maximum number of headlines appearing in a day (mh) in our training & test sample. We use padding to make all daily headline counts equal to this. As inputs we feed normal vectorised headlines through our pre trained individual headline encoder (FOOTNOTE: if the maximum headline count is 5 this means reusing our individual headline encoder 5 times. Using tensor flow we fix these weights - again due to sparsity of data -, choosing to focus on training the multiple headline encoder given these input instead.), reducing their dimensionality through pca whitening (NOTE: already outlined our aim of avoiding pca). In this way we pass mh encoded headlines through our max pooling RNN giving a 30 dimensional (same as input after pca) output vector. We can either train by linking this output to returns using a softmax function, or rather by attempting to decode the output (seq2seq as outlined previously using an RNN decoder). The later of these may involve one hot encoding headlines and using the cross entropy loss function, or better still, as already outlined, cosine similarity. 

IMPORTANT NOTE: if the training is linked to returns, only training data can be used!

IMPLEMENTATION: 
see Deep Learning seq2seq page again. 
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb

*suggestion to speed up training: in 2seq only output topstory and one or two randomly chosen encoded headlines.*       

*may need to use pca whitening in a different way - not sure if can be applied separately on each headline representation* - ISSUE ALREADY TACKLED - no pca  

Multiple Days - incorporates price and volume:
Using the pretrained network described in the previous section we analise 20 training days (1 month - roughly following Ding et al.) of headlines. As in the previous section, we fix the weights in the pretrained network, allowing us to pass individual headlines through the network (when we mention input in this section from here on, we mean the output from our pre-trained networks). Padding is used in instances where data is unavailable. 

To allow us to analyze headlines in conjunction with technical information, we concatenate two additional points to the network inputs. Elaborating to each day's headlines vector (input) we add the the days daily return and volume (could add first differences as well). This expands the input dimension to 32 (34 with first differences), spanning 20 day. Where padding is added for missing headlines, a vector of zeros with only price and volume data for that day is passed into the network. We follow the CNN encoder architecture outlined in Ding et al. (number of layers - convolution and max pooling -, step sizes, and channels/depth). 

NOTE: instead of concatenating simple/plain returns into the multi day network, z scores can be used, thus incorporating cross sectional information. Theory relating to the momentum factor considers this to be the causal factor (FOOTNOTE: the fact that something goes up in value is not important in predicting future trends. Rather the fact that something increases/decreases in value z times more than the mean is important in capturing price trends. This agrees with a paper summarised in the previous document.).

Ideally we would like to be able to train the network using an unsupervised encoder decoder framework.Such an approach is outlined in: http://mi.eng.cam.ac.uk/~cipolla/publications/article/2016-PAMI-SegNet.pdf  or  https://arxiv.org/pdf/1311.2901v3.pdf. A visualization of CNN decoder unsampling/deconvolution is available at: https://www.quora.com/How-do-fully-convolutional-networks-upsample-their-coarse-output. The approach functions as any encoder decoder framework thus far explained. The only complication arises when trying to understand the unsampling layer, which is why we provide a link showing the possible visualizations of this process.  

Despite this possible framework, due to data sparsity and the complexity of the approach, we decide to follow a simpler method requiring the training of only a CNN encoder. We are thus forced to link the network to returns, once again using a cross entropy loss function (we identify ys into buckets using these as classes for supervised training) to train network weights. 

IMPLEMENTATION: 
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
see Hands on Machine LEarning CovNet chapter 

Price and Volume Data: 

Perhaps the best way of considering price and volume data is to use the analogy of considering speech. Whilst auditory signals may be analysed on their own, better performance is registered when both auditory and visual signals are considered in conjunction. An efficient means of considering these and training an unsupervised network using an encoder and decoder framework are outlined in: https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf. Price and volume data con also be viewed separately; however as they are inherently linked (like the shape our mouths is with sound when pronouncing a syllable) considering these together should lead to better performance. We use LSTMs in Ngiam et al. 2011 framework, as this allows the continuation and adjustment of signals, whilst considering time effects. Combined with our handling of price and volume data through the CNN outlined in the previous section, this allows us to satisfy our aim of capturing the essence of technical analysis. 

Training follows the sequential and modular approach of Ngiam et al. We focus on a time frame of 20 training days (see note for expansion of this horizon) to coincide with data passed into our CNN. An important point to note is that by following the mentioned approach we learn a network that considers both inputs equally. This is achieved by training it to learn to replicate both price and volume data, when simply one form of input is provided. Once again we use a softmax loss function in this unsupervised training approach.  

*Note: may be able to incorporate momentum effect by adding 12 month historic returns before above mentioned 20 daily returns. This would be in accordance outlined in our previous summary.*

Bringing all modules together: 
Freezing weight in all pre trained modules (may have to unfreeze wights in modules trained using returns), we bring everything together by concatenating module encoder outputs (follow fusion layer in: http://ais.informatik.uni-freiburg.de/publications/papers/eitel15iros.pdf). To be more precise we concatenate outputs from the CNN as well as the price and volume joint encoder. Finally, to emphasize the current time period t, to this concatenation we also add the headlines vectors for today (output from one day headlines encoder using headlines from time t). This concatenated overall vector is then passed through a fully connected neural network with two layers (initially use best practice for setting number of network nodes - formula outlined in data mining coursework) using a ExC activation function (advanced ReLU with non zero gradient for negative values). As we freeze pre-trained weights we should be able to use cross validation to adjust the hyper parameters (enough data to do this considering we are only training a shallow layer).

As a final note: Training, and potential augmentation of our training size, can be achieved by randomly dropping input parameters; we can randomly zero out 50% of input parameters, forcing the network to better learn generalisable rules (suggested by audio and speech paper). Nevertheless, such an approach is only applicable when batch normalisation is used. This is further the case when padding is applied.

IMPLEMENTATION:
see Hands on Machine Learning page 317
see Deep Learning page 

Testing and Reinforcement Learning (OPTIONAL): 
Whilst we freeze the pre-trained weights in training our overall network, we consider the test sample as providing a small opportunity for adjusting these through reinforcement learning. Whilst any changes using a test sample of only roughly 2k headlines are going to be minimal, they offer a chance to improve our network nonetheless, without breaking any testing approaches. 

IMPLEMENTATION: 
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb

/IMPLEMENTATION OF ABOVE:/
Books: 
Fundamentals of Deep Learning - Nikhil Buduma
Hands on Machine Learning with Scikit-Learn and TensorFlow

Other References:
History: 
paper: http://site.iugaza.edu.ps/wdaya/files/2013/03/A-Random-Walk-Down-Wall-Street.pdf
summary: famously said that price movements resemble a random walk and are unpredictable. 

mention fama and french - debate about market efficiency culminating in 5 factor model: 
summary: reward is proportionate to risk. 

paper: http://efinance.org.cn/cn/fm/Does%20the%20Stock%20Market%20Overreact.pdf
summary: demonstrates behavioural inefficiencies and market's systemic overreaction to news 

Headlines & Natural Language Processing:
Mention story with Anne Hathaway and Berkshire Hathaway 
Overall aim: Trying to take one headline or possibly multiple headlines.... where each headline is represented as a point in an n dimensional space.... allowing us to identify clusters of headlines for which to buy, sell, hold..... provided we use a CNN on multiple points (and thus headlines) we attempt to identify the interactions (features) between them to once again identify clusters for buying, selling, and holding. The first step of this process involves representing a sentence of words as a single point..... from the work of Ding et al. as well as Richard Socher we have been led to believe that Recursive Neural Tensor Networks offer the best opportunity to do this. The next step either involves using a fully connected neural network to analyse one headline/point, or using CNNs multiple points... (CNNs become fully connected Ns after numerous max pooling procedures).

Paper 1: http://emnlp2014.org/papers/pdf/EMNLP2014148.pdf    
Summary: Among the first to use non linear approaches (neural nets) on structured data, instead of just linear models on bags of words to evaluate headlines. Argues that 'unstructured terms cannot indicate the actor and object of the event' thus poorly evaluating an events impact on stock price movements. Moreover, proves that NN approaches outperform previous simpler linear (eg linear SVM) attempts to analyse events. Overall the paper's method achieves 'accuracy of S&P 500 index prediction [of] 60%, and that of individual stock prediction [of]... over 70%'.  

Methodology:
Part 1a: Structured representation:
"Each event is composed of an action P, an actor O1 that conducted the action, and an object O2 on which the action was performed. Formally, an event is represented as E = (O1, P, O2, T), where P is the action, O1 is the actor, O2 is the object and T is the timestamp (T is mainly used for aligning stock data with news data)" 
As quoted above the authors use a (O,P,O,T) format to capture the information in headlines. They start by extracting the predicate type P, 'and then find the longest sequence of words Pv, such that Pv starts at P and satisfies the syntactic and lexical constraints proposed by Fader et al. (2011)' (paper: http://www.aclweb.org/anthology/D11-1142 - which describes open information extraction and dependency parsing). Next they identify O1, and O2; they find the nearest left and right noun phrases to P respectively (must contain subject and object otherwise headline is excluded). 
Part 1b:  Event Generalization: To reduce the number of (O,P,O) types the authors use WordNet to 'extract lemma forms of inflected words'. This translates eg verbs like 'changed' to 'change' and nouns like 'dogs' to 'dog'. After this VerbNet is used to tranche events; can be thought of as trying to put identified Ps into buckets representing events. 'After generalization, the event (Private sector, adds, 114,000 jobs) becomes (private sector, multiply_class, 114,000 job).'
Part 2: Prediction Models: The paper mentions SVMs as well as bag of words representation; however below we focus on their non linear approach implementing (O,P,O). Before continuing we must note that to avoid sparseness the paper transforms (O,P,O) into (O1, P, O2, O1 + P, P + O2, O1 + P + O2) - aka back-off features. The paper does not specify what method is used to transform the words appearing in (O,P,O) to vectors - I suspect they use the same TFIDF score as with bag of words: 'freq(tl) denote the number of occurrences of the lth word in the vocabulary in document d. TFl = 1 |d| freq(tl), ∀l ∈ [1 , L], where |d| is the number of words in the document d (stop words are removed). TFIDFl = 1 |d| freq(tl) × log( N |{d:freq(tl )>0}|), where N is the number of documents in the training set. The feature vector Φ can be represented as Φ = (ϕ1, ϕ2, ..., ϕM) = (TFIDF1 , TFIDF2 , ..., TFIDFM )'. Despite this, the authors mention using a two layer fully connected neural net (they try three layers but find no improvement in classification results) with two output classes (+1, -1) signifying buying and selling. Numerous Ys are used spanning three different horizons: one day, one week, and one month (with results showing diminishing results as the time horizon increases). The authors 'automatically align 1,782 instances of daily trading data with news titles and contents from the previous day' with headlines presumably being evaluated individually nonetheless. This implies that numerous headlines may share the same y in the authors paper. 

Improving on paper1: 
Paper 2: https://www.ijcai.org/Proceedings/15/Papers/329.pdf
Summary: At the moment the principal motivation to our own paper. The authors find an additional 6% classification accuracy on top of their previous paper (Paper 1) when analysing S&P500 returns. They achieve this improvement principally by focusing on deep learning (CNN architecture) whilst implementing word vectors (event embeddings, which are dense vectors) to represent their previous (O,P,O) format. Additionally, they examine multiple headlines at once, instead of just using one headline per y. 
Note: 'Embeddings are trained such that similar events, such as (Actor = Nvidia fourth quarter results, Action = miss, Object = views) and (Actor = Delta profit, Action = didn’t reach, Object = estimates), have similar vectors, even if they do not share common words.'
Methodology: 
Part 1: Event Representation and Extraction: 
They follow the same outline as in their first aper to reach (O,P,O). Nevertheless, the authors better specify their approach, detailing their use of 'ReVerb to extract the candidate tuples of the event (O 1 , P , O 2 ), and then parse the sentence with ZPar [Zhang and Clark, 2011] to extract the subject, object and predicate'. The same filtering is used as in paper one - if O1 and O2 do not contain subject and object the are filtered out. For absolute clarity we paste the extract from the first paper describing this below:
'For each event phrase Pv identified in the step above, we find the nearest noun phrase O1 to the left of Pv in the sentence, and O1 should contain the subject of the sentence (if it does not contain the subject of Pv, we find the second nearest noun phrase). Analogously, we find the nearest noun phrase O2 to the right of Pv in the sentence, and O2 should contain the object of the sentence (if it does not contain the object of Pv, we find the second nearest noun phrase).'
Part 2: Event Embedding (This is different to paper 1): 
Side Note:
Useful additional paper for understanding embedding: https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf - shows that 'performance can be improved when entities are represented as an average of their constituting word vectors'
video explaining above paper: https://www.youtube.com/watch?v=s1lzOkC2lxU
Side paper summary: essentially outlines the use of RNTNs to combine word vectors to get sentence word vectors (simplification).... they use this to identify probable relationships between words (e1 and e2).... since they have three words (e,r, e) the network must use two layers between (e, r) (r, e2) and (er, re2) to get a final score. 
further links to explain above: https://www.youtube.com/watch?v=mVfPGu8rrXM
Continuing with paper: We must note before describing the authors use of RNTNs that all phrases both for Os and the P exceeding one word are averaged (average word vectors - allowing you to represent Os like 'Nokia’s mobile phone business and Nokia'). The authors use the skip gram model implemented on a large financial corpus to learn these 100 dimensional word vectors. Moving on to describing the training, the authors then implement the same RNTN described above (two layers and etc). This is trained using margin loss and l2 regularisation - a corrupted OPO is created by replacing one O with a randomly selected word from the dictionary, with the network being taught to identify the true OPO by minimising the marginal loss. (500 iterations are conducted for every OPO). 
Part 3: Prediction Model: 
Because with one headline a fully connected network is used, I detail the authors use of CNNs over the medium and long term. Whilst the author does not specify how he tackles multiple headlines in any given day, examining the pictures provided it appears that for a horizon of 30 days he uses 30 inputs.... perhaps implying that he uses simple averaging of headlines? Regardless the structure involves sliding a window of size 3 over the data (presumably for both med and long term) repeatedly, and then max pooling the resulting layers over the depth (number of repeated passes). Afterwards the resulting layer is passed through a fully connected network. Interestingly before passing the short medium and long term through the fully connected layer, the resulting vectors are concatenated creating 'feature vectors V C = (V l , V m, V s )'. Subsequently a sigmoid activation function is used to determine a score for two classes (would be better to use softmax): buy sell [+1, -1]. 

Paper 3: 
paper: http://aisel.aisnet.org/cgi/viewcontent.cgi?article=1018&context=ecis2016_rip
summary: another paper examining deep learning and its use in single headline analysis. It compares the performance of a recursive autoencoder against a benchmark random forest approach finding a (roughly) 6 boost in performance resulting from implementing the deep network. 
Methodology: the benchmark follows a similar approach to the linear model in the first paper. A headline is prepossessed by removing punctuations and etc, before being translated into a text document matrix (measures how frequently each word appears in the training set using the probability for each word) before being used as part of a random forest. 
Moving on to the autoencoder: Potentially outlines a method for converting a headline into a single vector representation without using papers 1 and 2's OPO structured representation. Instead the paper notes that one hot vectors can be recursively combined to create encoded vectors - after the entire sentence has been passed through we have a vector encoding the entire sentence. NOT YET SURE HOW THIS IS TRAINED.....IF IT CAN BE DONE WITHOUT CLASSIFYING THE RESULTING VECTOR. 

Paper 4: 
outlined on github: https://github.com/WayneDW/Sentiment-Analysis-in-Event-Driven-Stock-Price-Movement-Prediction
Essentially converts headlines into word vector matrices (numerous words so matrix) which are then passed through a CNN that generates a buy sell hold signal for that headline. Justifies using CNNs by citing: http://www.aclweb.org/anthology/D14-1181 - this paper demonstrated the benefits of using CNNs in a variety of nlp tasks ranging from machine trans. to sentiment analysis. Further the author of the supplementary paper shows that not much training is required of the CNN weight parameters to achieve the positive results.  

SIDE NOTE FOR UNDERSTANDING THEORY:
To understand RNTN (Recurrent Neural Tensor Network) look at: 
https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
and 
https://www.youtube.com/watch?v=Z56jojdmDV0
start of with simple concatenation of word a and word b multiplied by a Tensor to give a combined representation.... this is then passed through an activation function that spits out a score. To train such a simple model you could change a to non sensical word (call it z) and look for a-b getting a larger score than z-b. To better capture the interaction between a-b, to the simple model a second procedure is added (to understand this just do the matrix multiplication) where a is multiplied by a tensor and then b to give one number in a vector of size d (the dimensionallity  of the original word vector)  

Technicals:
url: http://schwert.ssb.rochester.edu/f532/ff_JF08.pdf
from paper one but overall linked to above url: quote summarising why interested in technicals 'Indeed, Fama and French (2008) describe momentum as the “premier anomaly” in stock returns'

Paper 1: http://cs229.stanford.edu/proj2013/TakeuchiLee-ApplyingDeepLearningToEnhanceMomentumTradingStrategiesInStocks.pdf
summary: this paper discusses exploiting the momentum trading strategy (buying things that have gone up more than the industry average over the past 3 months and selling the etc) using a neural network architecture with 5 layers including the input layer. In short the paper finds that analysing price charts (t-3 to t-13 of monthly returns and past 20 days of daily returns) can in conjunction with a enhanced trading strategy 'deliver an annualised return of 45.93% over the 1990-2009 test period versus 10.53% for basic momentum'. 
Methodology: the paper uses a training sample from  January 1965 to December 1989 and a test sample up to 2009. The analyses is restricted  'to ordinary shares trading on NYSE, AMEX, or Nasdaq', with a closing monthly price higher than $5. Both monthly returns and daily returns are considered, with returns adjusted cross sectionally to z scores (like weighing a companies performance that month/day with respect to all other companies). This is done to capture the relevant information required for exploiting momentum trading.
Regarding the neural network architecture, whilst the author notes that he uses stacked restricted Boltzmann machines for encoding the original input (33 input parameters - concatenated monthly returns and daily returns) before passing resulting outputted results through a fully connected neural network, in essence the entire procedure may be viewed as a neural network with an expanding then contracting and finally expanding node structure ('e five-layer network 33−40−4−50−2 consisting of an encoder that takes 33 inputs and reduces them to a 4 dimensional code and a classifier that takes these 4 inputs and outputs the probabilities for the two classes'). 




