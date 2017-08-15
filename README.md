# Deep_Learning_for_Finance
MOST COMMON NEURAL NETWORK BUILDING BLOCKS DESCRIBED BELOW 
- Image summary available at: http://nyc2016.fpq.io/fpq_yam_peleg_deep_learning.pdf   page12

AIM AND BASICS DESCRIPTION:

Rather than try every neural architecture imaginable, we shall try and mimic best practices applied by investors daily. Hence, in the same way that convolutional neural networks (CNNs) mimic our own visual cortexes, our neural architecture shall mimic the discovery process followed by investors. Since we are focusing on technicals, as well as fundamentals appearing in textual form, we must attempt to find an architecture that looks for price chart patterns similarly as a day traded, and fundamentals as both an arbitrager and a longer term investor. Finally we must identify a method to combine these in an efficient way (see - https://people.csail.mit.edu/khosla/papers/icml2011_ngiam.pdf - for inspiration). 

Taking these in step, day traders (not considering event arbitragers yet) attempt to gauge the direction of the heard and profit from it. They are in affect comparable to cattle herders. In trying to fulfill there aim they primarily concentrate on identifying patters in price charts, looking for PATTERNS appearing in conjunction with the passage of TIME. Elaborating, a day trader may look for support and resistance lines (bands in which prices move), or heads and shoulders (a famous technical patter that some investors swear by), and make a trade based on his belief of the likelihood of this pattern continuing tomorrow, in the next hour or etc. The difficulty in mimicking this kind of analysis is that neither CNNs - which look for patterns and make decisions irrespective of where (time t or t-10) they occur -  nor recurrent neural networks - which try and analyse the cumulative effects of the passage of time, or addition of variables - on their own replicate this process completely. Ideally we would like an architecture that allows us to capture both the presence of patterns and the passage of time. Consequently, we choose to input price and volume data into two parts of our network - one CNN network, in conjunction with multiday 'headlines vectors', and one long short term memory RNN (LSTM). (see architecture section for more details)

*Other option: Consequently, we choose to implement a modern max pooling RNN (quote Michael's paper) when considering price and volume data (see architecture section for more details).*


*************************************************************
TECHNICAL ANALYSIS ADDITIONAL INFORMATION: 
also see paper on RSI and MACD 

Outline from my 2014 paper: 

Cesari and Cremonini (2003) suggested that technical analysis (TA) is the oldest investment appraisal technique used to beat markets. TA allows traders to evaluate the ‘breadth and duration of price trends’, providing evidence for imminent/immediate market movements before fundamental analysis (Doug Standefer, June 2003). It makes one assumption: that prices do not follow a markov process. 

Whilst many papers have been published rebuffing the claims of technical analysts, perhaps most prominently a paper by Fama and French, numerous other papers have been published providing support for the technique. Regardless, we set out to try and use price and volume data to distinguish between buy, sell, and hold points in our n dimensional field.

Traditional technical analysis, as you have discovered, focuses on using price and volume data as inputs in signal identifying functions. Elaborating, individual price data can be passed through functions such as the Moving Average Convergence Divergence function (better known as MACD) or Relative Strength Index function (RSI) (Note: see my crappy paper on technicals for more details if desired). These try and extract important trading signals from the seemingly random noise that is a price chart, allowing investors' to identify immediate trading signals (eg: when RSI dips above/below 70/30). These functions have numerous hyperparameters (eg. lenght of input data and etc), which are usually optimised over the training set. The problem with these techniques is that they tend to overfit on the training set, and rarely perform well over the test set. This is because no actual trading signal is identified in the process - the technique identifies esoteric relationships that would have worked in the past, however no real method for separating out noise is found. 

A plethora of other techniques exists, but rather than waste time outlining these I move on to our goal. Similarly to technical analysts we are taking price and volume data as inputs. The more information we use (adjusted closing price, open and close spread, volume, and etc - data downloaded from yahoo finance) the better our network should be able to identify clusters using technical information. However there is a difference between using more useful information, and using simplified, and possibly completely irrelevant, information. Expanding upon this point, using RSI or any other technical indicator is pointless as: all such information is present in the price chart already, the process is not supported by theory (no theory other than pure data mining), and it relies enormously on blindly  setting hyperparameters (data mining of training set with no link to test set). Whilst the inherent amount of noise and lack of data may make our task of identifying clusters using raw price and volume data impossible using a neural network, methods (supported by theory) for efficiently extracting information from price and volume data exist. These are the methods we shall focus on.

Before continuing, below we describe two observations linked to a so called momentum effect that researchers have tried to explain using risk factor theory: the principle that risk should be compensated by reward. The medium to long term momentum effect, in simple terms, notes that securities that have moved up in value more than the market average are likely to continue moving up in value, and those that have declined in value are more likely to continue declining in value. (FOOTNOTE: Requires observing a medium to long term horizon of returns, and typically continues for a period longer than a month.) The short term momentum effect (FOOTNOTE: better linked to the liquidity premium) on the contrary notes that companies that outperform the market today (in the short term) tend to underperform tomorrow (in the short term).

The reason that we spent space explaining the above, is that many researchers attribute the success of technical analysis to these phenomena. Moreover, an understanding of the above allows us to determine the data preprocessing required to capture real persistent trading signals without having to worry too much about hyperparameter (FOOTNOTE: whilst cross validation would still help identifying the optimal return horizon and etc, at least we have theoretical backing for our choice) s. We refer to paper 1 summarized in our outline (TECHNICAL's section) that described the use of NNs for effective 'technical' signal capturing. 

Per the paper mentioned, treating returns as panel data (both cross sectional and time series) allows us to capture an important trading signal. Elaborating, in every time period we can calculate the average and stdev daily/monthly return for all the companies being considered, and then use these to convert the returns to Z scores. Our model input would then consists of Z scores instead of returns. Similarly, to capture things such as the liquidity premium, we could calculate Z scores for volume as well. The above mentioned are two ways of 'actually extending' the data; unlike MA cross for instance.  

SUMMARY OF WHAT NEEDED FOR TECHNICALS:
DATA: 
Z score for price/return and volume
nominal volume (potentially allows for better individual security pattern spotting - heard behaviour)
adjusted close and adjusted open (adjusted open calculation outlined previously...... or just open close difference) 

TECHNIQUE ADDITIONALS: 
if focusing on next day return we shall only have to focus on 20 days of data as we shall be capturing the short term momentum effect. 
Otherwise have daily and monthly returns (like paper 1.... and all other data mentioned above) 


(Market forces: technically speaking…, Doug Standefer, June 2003)
**********************************************************************

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

RESULTS FOR TECHNICALS (MICHAELS's 20 DIM FILE0 USING AN CNN TO ENCODE TECHNICALS AND THEN CONCATENATING ENCODED RNN TEHCNICALS: 
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 11.0046s
Testing time: 0.0056s
Confusion matrix:
[[ 160  693   63]
 [  80 1625   48]
 [ 169  897   66]]
Accuracy: 0.4870
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 11.1451s
Testing time: 0.0023s
Confusion matrix:
[[ 160  694   62]
 [  79 1625   49]
 [ 169  896   67]]
Accuracy: 0.4872
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 101.0340s
Testing time: 0.0174s
Confusion matrix:
[[   0  915    1]
 [   0 1752    1]
 [   0 1131    1]]
Accuracy: 0.4612
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 104.4732s
Testing time: 0.0177s
Confusion matrix:
[[   0  915    1]
 [   0 1752    1]
 [   0 1131    1]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 13.8852s
Testing time: 0.0113s
Confusion matrix:
[[   4  912    0]
 [   0 1752    1]
 [   0 1131    1]]
Accuracy: 0.4622
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 25.0971s
Testing time: 0.0112s
Confusion matrix:
[[   0  916    0]
 [   1 1752    0]
 [   0 1131    1]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 31.0099s
Testing time: 0.0202s
Confusion matrix:
[[   2  914    0]
 [   0 1752    1]
 [   1 1131    0]]
Accuracy: 0.4615
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 61.1083s
Testing time: 0.0374s
Confusion matrix:
[[   1  915    0]
 [   0 1752    1]
 [   1 1131    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'SVM, adj.' classifier.
Classifier: SVM, adj.
Training time: 797.1659s
Testing time: 22.8669s
Confusion matrix:
[[  97  737   82]
 [  53 1659   41]
 [ 111  952   69]]
Accuracy: 0.4801
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 509.3368s
Testing time: 15.7483s
Confusion matrix:
[[  69  807   40]
 [  27 1707   19]
 [  71 1024   37]]
Accuracy: 0.4770
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 1.6142s
Testing time: 15.7253s
Confusion matrix:
[[359 355 202]
 [534 936 283]
 [409 481 242]]
Accuracy: 0.4044
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 1.9124s
Testing time: 0.0034s
Confusion matrix:
[[ 157  727   32]
 [  65 1656   32]
 [ 156  936   40]]
Accuracy: 0.4875
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 5.6458s
Testing time: 3.3196s
Confusion matrix:
[[ 169  597  150]
 [ 141 1472  140]
 [ 192  772  168]]
Accuracy: 0.4759
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1263s
Testing time: 3.3148s
Confusion matrix:
[[  49  863    4]
 [  20 1730    3]
 [  49 1078    5]]
Accuracy: 0.4694
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 22.2773s
Testing time: 0.1153s
Confusion matrix:
[[ 188  692   36]
 [ 109 1626   18]
 [ 200  894   38]]
Accuracy: 0.4872
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0262s
Testing time: 0.0080s
Confusion matrix:
[[ 105  708  103]
 [ 120 1562   71]
 [ 136  893  103]]
Accuracy: 0.4657
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.3618s
Testing time: 0.0024s
Confusion matrix:
[[ 164  683   69]
 [  85 1617   51]
 [ 176  885   71]]
Accuracy: 0.4872
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.4181s
Testing time: 0.0108s
Confusion matrix:
[[ 121  694  101]
 [  97 1571   85]
 [ 151  883   98]]
Accuracy: 0.4709
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">48.72%</td>
		<td style="text-align: right" >22.2773s</td>
		<td style="text-align: right" >0.1153s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger"><b>48.75%</b></td>
		<td style="text-align: right" >1.9124s</td>
		<td style="text-align: right" >0.0034s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">48.72%</td>
		<td style="text-align: right" >0.3618s</td>
		<td style="text-align: right" >0.0024s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">48.70%</td>
		<td style="text-align: right" >11.0046s</td>
		<td style="text-align: right" >0.0056s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger">48.72%</td>
		<td style="text-align: right" >11.1451s</td>
		<td style="text-align: right" ><b>0.0023s</b></td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">46.57%</td>
		<td style="text-align: right" >0.0262s</td>
		<td style="text-align: right" >0.0080s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">47.09%</td>
		<td style="text-align: right" >0.4181s</td>
		<td style="text-align: right" >0.0108s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">46.22%</td>
		<td style="text-align: right" >13.8852s</td>
		<td style="text-align: right" >0.0113s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >25.0971s</td>
		<td style="text-align: right" >0.0112s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >101.0340s</td>
		<td style="text-align: right" >0.0174s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >104.4732s</td>
		<td style="text-align: right" >0.0177s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">46.15%</td>
		<td style="text-align: right" >31.0099s</td>
		<td style="text-align: right" >0.0202s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >61.1083s</td>
		<td style="text-align: right" >0.0374s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">47.59%</td>
		<td style="text-align: right" >5.6458s</td>
		<td style="text-align: right" >3.3196s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">46.94%</td>
		<td style="text-align: right" >0.1263s</td>
		<td style="text-align: right" >3.3148s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger">48.01%</td>
		<td style="text-align: right" >797.1659s</td>
		<td style="text-align: right" class="danger">22.8669s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">47.70%</td>
		<td style="text-align: right" >509.3368s</td>
		<td style="text-align: right" class="danger">15.7483s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">40.44%</td>
		<td style="text-align: right" >1.6142s</td>
		<td style="text-align: right" class="danger">15.7253s</td>
    </tr>
</tbody>
</table>

RESULTS USING JUST 20DIM ENCODED TECHNICALS:
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 0.8564s
Testing time: 0.0072s
Confusion matrix:
[[ 170  683   63]
 [  87 1626   40]
 [ 166  895   71]]
Accuracy: 0.4912
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 1.0104s
Testing time: 0.0015s
Confusion matrix:
[[ 170  683   63]
 [  87 1626   40]
 [ 166  895   71]]
Accuracy: 0.4912
################################################################################
Start fitting 'RBM 200, n_iter=40, LR=0.01, Reg: C=1' classifier.
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 74.9017s
Testing time: 0.0182s
Confusion matrix:
[[   0  916    0]
 [   0 1752    1]
 [   0 1132    0]]
Accuracy: 0.4609
################################################################################
Start fitting 'RBM 200, n_iter=40, LR=0.01, Reg: C=10000' classifier.
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 75.5168s
Testing time: 0.0181s
Confusion matrix:
[[   0  915    1]
 [   0 1752    1]
 [   0 1131    1]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 9.0881s
Testing time: 0.0106s
Confusion matrix:
[[   0  915    1]
 [   0 1753    0]
 [   1 1131    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 16.9482s
Testing time: 0.0105s
Confusion matrix:
[[   0  916    0]
 [   0 1752    1]
 [   0 1132    0]]
Accuracy: 0.4609
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 17.1725s
Testing time: 0.0216s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 31.0654s
Testing time: 0.0415s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'SVM, adj.' classifier.

Classifier: SVM, adj.
Training time: 229.2236s
Testing time: 6.5703s
Confusion matrix:
[[ 127  748   41]
 [  49 1685   19]
 [ 119  973   40]]
Accuracy: 0.4872
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 118.8851s
Testing time: 2.7688s
Confusion matrix:
[[  90  813   13]
 [  30 1716    7]
 [  89 1030   13]]
Accuracy: 0.4786
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 0.0575s
Testing time: 1.1715s
Confusion matrix:
[[359 352 205]
 [528 957 268]
 [408 502 222]]
Accuracy: 0.4046
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 0.5653s
Testing time: 0.0017s
Confusion matrix:
[[ 126  723   67]
 [  53 1657   43]
 [ 126  932   74]]
Accuracy: 0.4886
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 3.4770s
Testing time: 3.0867s
Confusion matrix:
[[ 166  600  150]
 [ 142 1448  163]
 [ 194  773  165]]
Accuracy: 0.4680
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1120s
Testing time: 3.1413s
Confusion matrix:
[[  95  776   45]
 [  45 1686   22]
 [ 101  989   42]]
Accuracy: 0.4796
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 6.7727s
Testing time: 0.1176s
Confusion matrix:
[[ 145  702   69]
 [  89 1628   36]
 [ 164  898   70]]
Accuracy: 0.4849
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0160s
Testing time: 0.0051s
Confusion matrix:
[[ 282  546   88]
 [ 262 1382  109]
 [ 338  683  111]]
Accuracy: 0.4670
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.0666s
Testing time: 0.0014s
Confusion matrix:
[[ 173  680   63]
 [  93 1613   47]
 [ 170  881   81]]
Accuracy: 0.4912
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.0344s
Testing time: 0.0053s
Confusion matrix:
[[ 164  645  107]
 [ 105 1559   89]
 [ 186  851   95]]
Accuracy: 0.4783
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">48.49%</td>
		<td style="text-align: right" >6.7727s</td>
		<td style="text-align: right" >0.1176s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger">48.86%</td>
		<td style="text-align: right" >0.5653s</td>
		<td style="text-align: right" >0.0017s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger"><b>49.12%</b></td>
		<td style="text-align: right" >0.0666s</td>
		<td style="text-align: right" ><b>0.0014s</b></td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger"><b>49.12%</b></td>
		<td style="text-align: right" >0.8564s</td>
		<td style="text-align: right" >0.0072s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger"><b>49.12%</b></td>
		<td style="text-align: right" >1.0104s</td>
		<td style="text-align: right" >0.0015s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">46.70%</td>
		<td style="text-align: right" >0.0160s</td>
		<td style="text-align: right" >0.0051s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">47.83%</td>
		<td style="text-align: right" >0.0344s</td>
		<td style="text-align: right" >0.0053s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >9.0881s</td>
		<td style="text-align: right" >0.0106s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">46.09%</td>
		<td style="text-align: right" >16.9482s</td>
		<td style="text-align: right" >0.0105s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">46.09%</td>
		<td style="text-align: right" >74.9017s</td>
		<td style="text-align: right" >0.0182s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >75.5168s</td>
		<td style="text-align: right" >0.0181s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >17.1725s</td>
		<td style="text-align: right" >0.0216s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >31.0654s</td>
		<td style="text-align: right" >0.0415s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">46.80%</td>
		<td style="text-align: right" >3.4770s</td>
		<td style="text-align: right" >3.0867s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">47.96%</td>
		<td style="text-align: right" >0.1120s</td>
		<td style="text-align: right" >3.1413s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger">48.72%</td>
		<td style="text-align: right" >229.2236s</td>
		<td style="text-align: right" class="danger">6.5703s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">47.86%</td>
		<td style="text-align: right" >118.8851s</td>
		<td style="text-align: right" >2.7688s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">40.46%</td>
		<td style="text-align: right" >0.0575s</td>
		<td style="text-align: right" >1.1715s</td>
    </tr>
</tbody>
</table>
RESULTS USING RNN AS ABOVE 20 DIM NO HEADLINES OR CNN BUT FOR T+2
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 0.7637s
Testing time: 0.0016s
Confusion matrix:
[[179 486 419]
 [ 96 842 339]
 [204 671 565]]
Accuracy: 0.4173
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 0.7485s
Testing time: 0.0015s
Confusion matrix:
[[179 486 419]
 [ 96 842 339]
 [204 671 565]]
Accuracy: 0.4173
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 78.1502s
Testing time: 0.0183s
Confusion matrix:
[[  1 564 519]
 [  1 854 422]
 [  0 762 678]]
Accuracy: 0.4033
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 78.2110s
Testing time: 0.0183s
Confusion matrix:
[[  0 564 520]
 [  1 854 422]
 [  0 763 677]]
Accuracy: 0.4028
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 9.1680s
Testing time: 0.0108s
Confusion matrix:
[[  1 565 518]
 [  1 853 423]
 [  0 761 679]]
Accuracy: 0.4033
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 17.1364s
Testing time: 0.0108s
Confusion matrix:
[[  1 564 519]
 [  0 855 422]
 [  1 762 677]]
Accuracy: 0.4033
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 17.7424s
Testing time: 0.0219s
Confusion matrix:
[[  0 564 520]
 [  0 853 424]
 [  0 761 679]]
Accuracy: 0.4031
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 31.0064s
Testing time: 0.0418s
Confusion matrix:
[[  1 564 519]
 [  1 853 423]
 [  0 761 679]]
Accuracy: 0.4033
################################################################################
Start fitting 'SVM, adj.' classifier.

Classifier: SVM, adj.
Training time: 226.1747s
Testing time: 7.6812s
Confusion matrix:
[[124 385 575]
 [ 59 758 460]
 [125 582 733]]
Accuracy: 0.4249
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 124.9017s
Testing time: 3.2273s
Confusion matrix:
[[ 81 530 473]
 [ 30 871 376]
 [ 77 708 655]]
Accuracy: 0.4228
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 0.0524s
Testing time: 1.1970s
Confusion matrix:
[[490 269 325]
 [521 419 337]
 [680 363 397]]
Accuracy: 0.3436
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 0.5682s
Testing time: 0.0017s
Confusion matrix:
[[187 288 609]
 [ 90 615 572]
 [226 420 794]]
Accuracy: 0.4199
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 3.5464s
Testing time: 3.0851s
Confusion matrix:
[[344 343 397]
 [293 597 387]
 [451 485 504]]
Accuracy: 0.3802
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1178s
Testing time: 3.1371s
Confusion matrix:
[[150 410 524]
 [ 69 763 445]
 [187 594 659]]
Accuracy: 0.4136
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 6.7311s
Testing time: 0.1222s
Confusion matrix:
[[186 365 533]
 [102 737 438]
 [233 551 656]]
Accuracy: 0.4154
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0156s
Testing time: 0.0052s
Confusion matrix:
[[259 631 194]
 [150 945 182]
 [319 847 274]]
Accuracy: 0.3888
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.0511s
Testing time: 0.0015s
Confusion matrix:
[[185 458 441]
 [ 95 823 359]
 [211 637 592]]
Accuracy: 0.4209
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.0350s
Testing time: 0.0056s
Confusion matrix:
[[ 189  743  152]
 [  99 1063  115]
 [ 203 1016  221]]
Accuracy: 0.3875
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">41.54%</td>
		<td style="text-align: right" >6.7311s</td>
		<td style="text-align: right" >0.1222s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger">41.99%</td>
		<td style="text-align: right" >0.5682s</td>
		<td style="text-align: right" >0.0017s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">42.09%</td>
		<td style="text-align: right" >0.0511s</td>
		<td style="text-align: right" ><b>0.0015s</b></td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">41.73%</td>
		<td style="text-align: right" >0.7637s</td>
		<td style="text-align: right" >0.0016s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger">41.73%</td>
		<td style="text-align: right" >0.7485s</td>
		<td style="text-align: right" >0.0015s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">38.88%</td>
		<td style="text-align: right" >0.0156s</td>
		<td style="text-align: right" >0.0052s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">38.75%</td>
		<td style="text-align: right" >0.0350s</td>
		<td style="text-align: right" >0.0056s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">40.33%</td>
		<td style="text-align: right" >9.1680s</td>
		<td style="text-align: right" >0.0108s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">40.33%</td>
		<td style="text-align: right" >17.1364s</td>
		<td style="text-align: right" >0.0108s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">40.33%</td>
		<td style="text-align: right" >78.1502s</td>
		<td style="text-align: right" >0.0183s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">40.28%</td>
		<td style="text-align: right" >78.2110s</td>
		<td style="text-align: right" >0.0183s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">40.31%</td>
		<td style="text-align: right" >17.7424s</td>
		<td style="text-align: right" >0.0219s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">40.33%</td>
		<td style="text-align: right" >31.0064s</td>
		<td style="text-align: right" >0.0418s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">38.02%</td>
		<td style="text-align: right" >3.5464s</td>
		<td style="text-align: right" >3.0851s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">41.36%</td>
		<td style="text-align: right" >0.1178s</td>
		<td style="text-align: right" >3.1371s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger"><b>42.49%</b></td>
		<td style="text-align: right" >226.1747s</td>
		<td style="text-align: right" class="danger">7.6812s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">42.28%</td>
		<td style="text-align: right" >124.9017s</td>
		<td style="text-align: right" >3.2273s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">34.36%</td>
		<td style="text-align: right" >0.0524s</td>
		<td style="text-align: right" >1.1970s</td>
    </tr>
</tbody>
</table>

RESULTS FOR T+3 (ABOVE RNN NO HEADLINE) 
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 0.7755s
Testing time: 0.0015s
Confusion matrix:
[[ 160  142  908]
 [  45  294  645]
 [ 160  211 1236]]
Accuracy: 0.4446
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 0.7698s
Testing time: 0.0015s
Confusion matrix:
[[ 160  142  908]
 [  45  294  645]
 [ 160  211 1236]]
Accuracy: 0.4446
################################################################################
Start fitting 'RBM 200, n_iter=40, LR=0.01, Reg: C=1' classifier.
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 78.6890s
Testing time: 0.0186s
Confusion matrix:
[[   1    0 1209]
 [   1    0  983]
 [   1    0 1606]]
Accuracy: 0.4228
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 79.4554s
Testing time: 0.0181s
Confusion matrix:
[[   1    1 1208]
 [   0    0  984]
 [   0    1 1606]]
Accuracy: 0.4228
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 9.1441s
Testing time: 0.0107s
Confusion matrix:
[[   0    1 1209]
 [   0    1  983]
 [   0    0 1607]]
Accuracy: 0.4230
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 17.1865s
Testing time: 0.0106s
Confusion matrix:
[[   2    0 1208]
 [   0    1  983]
 [   0    0 1607]]
Accuracy: 0.4236
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 17.6313s
Testing time: 0.0218s
Confusion matrix:
[[   1    0 1209]
 [   1    0  983]
 [   0    0 1607]]
Accuracy: 0.4230
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 30.4648s
Testing time: 0.0417s
Confusion matrix:
[[   0    0 1210]
 [   0    0  984]
 [   0    0 1607]]
Accuracy: 0.4228
################################################################################
Start fitting 'SVM, adj.' classifier.

Classifier: SVM, adj.
Training time: 223.5923s
Testing time: 7.7022s
Confusion matrix:
[[  88   88 1034]
 [  19  224  741]
 [  88  128 1391]]
Accuracy: 0.4480
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 110.7594s
Testing time: 3.3264s
Confusion matrix:
[[   0    0 1210]
 [   0    0  984]
 [   0    0 1607]]
Accuracy: 0.4228
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 0.0530s
Testing time: 1.1735s
Confusion matrix:
[[576 199 435]
 [447 264 273]
 [802 277 528]]
Accuracy: 0.3599
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 0.5701s
Testing time: 0.0017s
Confusion matrix:
[[  84  115 1011]
 [  58  225  701]
 [ 115  150 1342]]
Accuracy: 0.4344
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 3.6458s
Testing time: 3.0965s
Confusion matrix:
[[447 182 581]
 [305 288 391]
 [600 259 748]]
Accuracy: 0.3902
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1296s
Testing time: 3.1275s
Confusion matrix:
[[ 102  117  991]
 [  30  242  712]
 [  93  154 1360]]
Accuracy: 0.4483
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 6.7980s
Testing time: 0.1151s
Confusion matrix:
[[ 173  120  917]
 [  54  251  679]
 [ 199  158 1250]]
Accuracy: 0.4404
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0158s
Testing time: 0.0051s
Confusion matrix:
[[266 665 279]
 [ 87 706 191]
 [318 888 401]]
Accuracy: 0.3612
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.0525s
Testing time: 0.0014s
Confusion matrix:
[[ 170  143  897]
 [  52  299  633]
 [ 173  215 1219]]
Accuracy: 0.4441
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.0341s
Testing time: 0.0054s
Confusion matrix:
[[ 204  747  259]
 [  56  789  139]
 [ 225 1047  335]]
Accuracy: 0.3494
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">44.04%</td>
		<td style="text-align: right" >6.7980s</td>
		<td style="text-align: right" >0.1151s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger">43.44%</td>
		<td style="text-align: right" >0.5701s</td>
		<td style="text-align: right" >0.0017s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">44.41%</td>
		<td style="text-align: right" >0.0525s</td>
		<td style="text-align: right" ><b>0.0014s</b></td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">44.46%</td>
		<td style="text-align: right" >0.7755s</td>
		<td style="text-align: right" >0.0015s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger">44.46%</td>
		<td style="text-align: right" >0.7698s</td>
		<td style="text-align: right" >0.0015s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">36.12%</td>
		<td style="text-align: right" >0.0158s</td>
		<td style="text-align: right" >0.0051s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">34.94%</td>
		<td style="text-align: right" >0.0341s</td>
		<td style="text-align: right" >0.0054s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">42.30%</td>
		<td style="text-align: right" >9.1441s</td>
		<td style="text-align: right" >0.0107s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">42.36%</td>
		<td style="text-align: right" >17.1865s</td>
		<td style="text-align: right" >0.0106s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">42.28%</td>
		<td style="text-align: right" >78.6890s</td>
		<td style="text-align: right" >0.0186s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">42.28%</td>
		<td style="text-align: right" >79.4554s</td>
		<td style="text-align: right" >0.0181s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">42.30%</td>
		<td style="text-align: right" >17.6313s</td>
		<td style="text-align: right" >0.0218s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">42.28%</td>
		<td style="text-align: right" >30.4648s</td>
		<td style="text-align: right" >0.0417s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">39.02%</td>
		<td style="text-align: right" >3.6458s</td>
		<td style="text-align: right" >3.0965s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger"><b>44.83%</b></td>
		<td style="text-align: right" >0.1296s</td>
		<td style="text-align: right" >3.1275s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger">44.80%</td>
		<td style="text-align: right" >223.5923s</td>
		<td style="text-align: right" class="danger">7.7022s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">42.28%</td>
		<td style="text-align: right" >110.7594s</td>
		<td style="text-align: right" >3.3264s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">35.99%</td>
		<td style="text-align: right" >0.0530s</td>
		<td style="text-align: right" >1.1735s</td>
    </tr>
</tbody>
</table>

SAME AS ABOVE BUT T+4
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 0.7235s
Testing time: 0.0015s
Confusion matrix:
[[ 118   29 1116]
 [  27   91  720]
 [ 111   45 1544]]
Accuracy: 0.4612
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 0.7153s
Testing time: 0.0016s
Confusion matrix:
[[ 118   29 1116]
 [  27   91  720]
 [ 111   45 1544]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 200, n_iter=40, LR=0.01, Reg: C=1' classifier.
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 74.4200s
Testing time: 0.0180s
Confusion matrix:
[[   2    0 1261]
 [   1    0  837]
 [   0    1 1699]]
Accuracy: 0.4475
################################################################################
Start fitting 'RBM 200, n_iter=40, LR=0.01, Reg: C=10000' classifier.
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 75.5303s
Testing time: 0.0181s
Confusion matrix:
[[   2    0 1261]
 [   0    1  837]
 [   0    1 1699]]
Accuracy: 0.4478
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 9.0648s
Testing time: 0.0106s
Confusion matrix:
[[   0    1 1262]
 [   1    0  837]
 [   0    0 1700]]
Accuracy: 0.4473
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 16.8721s
Testing time: 0.0106s
Confusion matrix:
[[   2    0 1261]
 [   0    0  838]
 [   0    0 1700]]
Accuracy: 0.4478
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 17.5055s
Testing time: 0.0217s
Confusion matrix:
[[   1    0 1262]
 [   1    0  837]
 [   0    0 1700]]
Accuracy: 0.4475
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 30.6119s
Testing time: 0.0412s
Confusion matrix:
[[   0    0 1263]
 [   0    0  838]
 [   0    0 1700]]
Accuracy: 0.4473
################################################################################
Start fitting 'SVM, adj.' classifier.

Classifier: SVM, adj.
Training time: 225.2418s
Testing time: 7.7236s
Confusion matrix:
[[  62   28 1173]
 [  16  101  721]
 [  53   42 1605]]
Accuracy: 0.4651
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 98.8414s
Testing time: 3.3067s
Confusion matrix:
[[   0    0 1263]
 [   0    0  838]
 [   0    0 1700]]
Accuracy: 0.4473
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 0.0525s
Testing time: 1.1680s
Confusion matrix:
[[622 168 473]
 [405 163 270]
 [864 211 625]]
Accuracy: 0.3710
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 0.5771s
Testing time: 0.0016s
Confusion matrix:
[[ 144   44 1075]
 [  69  134  635]
 [ 183   83 1434]]
Accuracy: 0.4504
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 3.6430s
Testing time: 3.0854s
Confusion matrix:
[[491  96 676]
 [270 170 398]
 [673 141 886]]
Accuracy: 0.4070
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1301s
Testing time: 3.1380s
Confusion matrix:
[[  52   56 1155]
 [  13  144  681]
 [  52   86 1562]]
Accuracy: 0.4625
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 6.7915s
Testing time: 0.1157s
Confusion matrix:
[[ 179   51 1033]
 [  55  132  651]
 [ 212   75 1413]]
Accuracy: 0.4536
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0154s
Testing time: 0.0050s
Confusion matrix:
[[278 666 319]
 [ 91 576 171]
 [316 919 465]]
Accuracy: 0.3470
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.0503s
Testing time: 0.0014s
Confusion matrix:
[[ 132   39 1092]
 [  29  106  703]
 [ 118   61 1521]]
Accuracy: 0.4628
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.0351s
Testing time: 0.0053s
Confusion matrix:
[[ 214  762  287]
 [  61  646  131]
 [ 228 1074  398]]
Accuracy: 0.3310
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">45.36%</td>
		<td style="text-align: right" >6.7915s</td>
		<td style="text-align: right" >0.1157s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger">45.04%</td>
		<td style="text-align: right" >0.5771s</td>
		<td style="text-align: right" >0.0016s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">46.28%</td>
		<td style="text-align: right" >0.0503s</td>
		<td style="text-align: right" ><b>0.0014s</b></td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >0.7235s</td>
		<td style="text-align: right" >0.0015s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >0.7153s</td>
		<td style="text-align: right" >0.0016s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">34.70%</td>
		<td style="text-align: right" >0.0154s</td>
		<td style="text-align: right" >0.0050s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">33.10%</td>
		<td style="text-align: right" >0.0351s</td>
		<td style="text-align: right" >0.0053s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">44.73%</td>
		<td style="text-align: right" >9.0648s</td>
		<td style="text-align: right" >0.0106s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">44.78%</td>
		<td style="text-align: right" >16.8721s</td>
		<td style="text-align: right" >0.0106s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">44.75%</td>
		<td style="text-align: right" >74.4200s</td>
		<td style="text-align: right" >0.0180s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">44.78%</td>
		<td style="text-align: right" >75.5303s</td>
		<td style="text-align: right" >0.0181s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">44.75%</td>
		<td style="text-align: right" >17.5055s</td>
		<td style="text-align: right" >0.0217s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">44.73%</td>
		<td style="text-align: right" >30.6119s</td>
		<td style="text-align: right" >0.0412s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">40.70%</td>
		<td style="text-align: right" >3.6430s</td>
		<td style="text-align: right" >3.0854s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">46.25%</td>
		<td style="text-align: right" >0.1301s</td>
		<td style="text-align: right" >3.1380s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger"><b>46.51%</b></td>
		<td style="text-align: right" >225.2418s</td>
		<td style="text-align: right" class="danger">7.7236s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">44.73%</td>
		<td style="text-align: right" >98.8414s</td>
		<td style="text-align: right" >3.3067s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">37.10%</td>
		<td style="text-align: right" >0.0525s</td>
		<td style="text-align: right" >1.1680s</td>
    </tr>
</tbody>
</table>


RESULTS USING KRYSTOF 15 DIM ENCODED TRECHNICALS - NO CNN
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 0.8292s
Testing time: 0.0212s
Confusion matrix:
[[   1  915    0]
 [   1 1752    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 1.5541s
Testing time: 0.0014s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 80.8919s
Testing time: 0.0200s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 81.0430s
Testing time: 0.0205s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 8.6977s
Testing time: 0.0108s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 16.6849s
Testing time: 0.0108s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 15.9381s
Testing time: 0.0226s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.
Classifier: RBM 512, n_iter=100
Training time: 28.2213s
Testing time: 0.0419s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'SVM, adj.' classifier.

Classifier: SVM, adj.
Training time: 187.9059s
Testing time: 6.2480s
Confusion matrix:
[[   0  916    0]
 [   1 1752    0]
 [   3 1129    0]]
Accuracy: 0.4609
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 78.1818s
Testing time: 2.3537s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 0.0474s
Testing time: 0.0502s
Confusion matrix:
[[342 404 170]
 [645 792 316]
 [446 469 217]]
Accuracy: 0.3554
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 0.4262s
Testing time: 0.0017s
Confusion matrix:
[[  24  847   45]
 [  17 1697   39]
 [  35 1048   49]]
Accuracy: 0.4657
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 4.0751s
Testing time: 3.0953s
Confusion matrix:
[[200 508 208]
 [382 991 380]
 [252 624 256]]
Accuracy: 0.3807
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1278s
Testing time: 3.1354s
Confusion matrix:
[[  29  884    3]
 [  24 1728    1]
 [  29 1102    1]]
Accuracy: 0.4625
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 5.1463s
Testing time: 0.1178s
Confusion matrix:
[[  33  868   15]
 [  28 1715   10]
 [  47 1075   10]]
Accuracy: 0.4625
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.0136s
Testing time: 0.0048s
Confusion matrix:
[[  69   11  836]
 [  61   31 1661]
 [  86   30 1016]]
Accuracy: 0.2936
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 0.0630s
Testing time: 0.0013s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 0.0271s
Testing time: 0.0050s
Confusion matrix:
[[  31  211  674]
 [  29  562 1162]
 [  50  246  836]]
Accuracy: 0.3760
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">46.25%</td>
		<td style="text-align: right" >5.1463s</td>
		<td style="text-align: right" >0.1178s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger"><b>46.57%</b></td>
		<td style="text-align: right" >0.4262s</td>
		<td style="text-align: right" >0.0017s</td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >0.0630s</td>
		<td style="text-align: right" ><b>0.0013s</b></td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >0.8292s</td>
		<td style="text-align: right" >0.0212s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >1.5541s</td>
		<td style="text-align: right" >0.0014s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">29.36%</td>
		<td style="text-align: right" >0.0136s</td>
		<td style="text-align: right" >0.0048s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">37.60%</td>
		<td style="text-align: right" >0.0271s</td>
		<td style="text-align: right" >0.0050s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >8.6977s</td>
		<td style="text-align: right" >0.0108s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >16.6849s</td>
		<td style="text-align: right" >0.0108s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >80.8919s</td>
		<td style="text-align: right" >0.0200s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >81.0430s</td>
		<td style="text-align: right" >0.0205s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >15.9381s</td>
		<td style="text-align: right" >0.0226s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >28.2213s</td>
		<td style="text-align: right" >0.0419s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">38.07%</td>
		<td style="text-align: right" >4.0751s</td>
		<td style="text-align: right" >3.0953s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">46.25%</td>
		<td style="text-align: right" >0.1278s</td>
		<td style="text-align: right" >3.1354s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger">46.09%</td>
		<td style="text-align: right" >187.9059s</td>
		<td style="text-align: right" class="danger">6.2480s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >78.1818s</td>
		<td style="text-align: right" >2.3537s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">35.54%</td>
		<td style="text-align: right" >0.0474s</td>
		<td style="text-align: right" >0.0502s</td>
    </tr>
</tbody>
</table>

RESULTS USING CONCATENATED INDIVIDUAL 300 DIM MT HEADLINES WITH ENCODED 20 DIM TECHNICALS
Got 60242 training samples and 3801 test samples.
################################################################################
Start fitting 'Logistic Regression (C=1)' classifier.
Classifier: Logistic Regression (C=1)
Training time: 70.3083s
Testing time: 0.0059s
Confusion matrix:
[[ 116  678  122]
 [  54 1611   88]
 [ 121  883  128]]
Accuracy: 0.4880
################################################################################
Start fitting 'Logistic Regression (C=1000)' classifier.
Classifier: Logistic Regression (C=1000)
Training time: 192.9124s
Testing time: 0.0040s
Confusion matrix:
[[ 126  678  112]
 [  60 1605   88]
 [ 136  866  130]]
Accuracy: 0.4896
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=1
Training time: 218.7416s
Testing time: 0.0232s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Classifier: RBM 200, n_iter=40, LR=0.01, Reg: C=10000
Training time: 218.5600s
Testing time: 0.0235s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100' classifier.
Classifier: RBM 100
Training time: 31.7963s
Testing time: 0.0137s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 100, n_iter=20' classifier.
Classifier: RBM 100, n_iter=20
Training time: 62.9672s
Testing time: 0.0137s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 256' classifier.
Classifier: RBM 256
Training time: 75.7329s
Testing time: 0.0276s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'RBM 512, n_iter=100' classifier.

Classifier: RBM 512, n_iter=100
Training time: 146.0882s
Testing time: 0.0520s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'SVM, adj.' classifier.
Classifier: SVM, adj.
Training time: 1661.0278s
Testing time: 56.2758s
Confusion matrix:
[[  94  732   90]
 [  40 1658   55]
 [ 104  945   83]]
Accuracy: 0.4828
################################################################################
Start fitting 'SVM, linear' classifier.
Classifier: SVM, linear
Training time: 1191.7719s
Testing time: 47.0907s
Confusion matrix:
[[  78  813   25]
 [  27 1713   13]
 [  75 1030   27]]
Accuracy: 0.4783
################################################################################
Start fitting 'k nn' classifier.
Classifier: k nn
Training time: 1.2344s
Testing time: 65.0951s
Confusion matrix:
[[375 353 188]
 [578 885 290]
 [420 488 224]]
Accuracy: 0.3904
################################################################################
Start fitting 'Decision Tree' classifier.
Classifier: Decision Tree
Training time: 8.9846s
Testing time: 0.0029s
Confusion matrix:
[[ 145  735   36]
 [  77 1641   35]
 [ 140  947   45]]
Accuracy: 0.4817
################################################################################
Start fitting 'Random Forest' classifier.
Classifier: Random Forest
Training time: 16.1065s
Testing time: 3.1128s
Confusion matrix:
[[ 145  641  130]
 [ 117 1510  126]
 [ 165  835  132]]
Accuracy: 0.4701
################################################################################
Start fitting 'Random Forest 2' classifier.
Classifier: Random Forest 2
Training time: 0.1387s
Testing time: 3.1423s
Confusion matrix:
[[   0  916    0]
 [   0 1753    0]
 [   0 1132    0]]
Accuracy: 0.4612
################################################################################
Start fitting 'AdaBoost' classifier.
Classifier: AdaBoost
Training time: 98.5624s
Testing time: 0.1409s
Confusion matrix:
[[ 188  696   32]
 [ 118 1622   13]
 [ 212  901   19]]
Accuracy: 0.4812
################################################################################
Start fitting 'Naive Bayes' classifier.
Classifier: Naive Bayes
Training time: 0.1988s
Testing time: 0.0224s
Confusion matrix:
[[ 286  557   73]
 [ 538 1170   45]
 [ 347  688   97]]
Accuracy: 0.4086
################################################################################
Start fitting 'LDA' classifier.
Classifier: LDA
Training time: 2.6718s
Testing time: 0.0044s
Confusion matrix:
[[ 129  673  114]
 [  64 1594   95]
 [ 147  848  137]]
Accuracy: 0.4893
################################################################################
Start fitting 'QDA' classifier.
Classifier: QDA
Training time: 1.9541s
Testing time: 0.0483s
Confusion matrix:
[[ 184  523  209]
 [ 226 1256  271]
 [ 195  695  242]]
Accuracy: 0.4425
<table class="table">
  <thead>
    <tr>
        <th>Classifier</th>
        <th>Accuracy</th>
        <th>Training Time</th>
        <th>Testing Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
		<td>AdaBoost</td>
		<td style="text-align: right" class="danger">48.12%</td>
		<td style="text-align: right" >98.5624s</td>
		<td style="text-align: right" >0.1409s</td>
    </tr>
    <tr>
		<td>Decision Tree</td>
		<td style="text-align: right" class="danger">48.17%</td>
		<td style="text-align: right" >8.9846s</td>
		<td style="text-align: right" ><b>0.0029s</b></td>
    </tr>
    <tr>
		<td>LDA</td>
		<td style="text-align: right" class="danger">48.93%</td>
		<td style="text-align: right" >2.6718s</td>
		<td style="text-align: right" >0.0044s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1)</td>
		<td style="text-align: right" class="danger">48.80%</td>
		<td style="text-align: right" >70.3083s</td>
		<td style="text-align: right" >0.0059s</td>
    </tr>
    <tr>
		<td>Logistic Regression (C=1000)</td>
		<td style="text-align: right" class="danger"><b>48.96%</b></td>
		<td style="text-align: right" >192.9124s</td>
		<td style="text-align: right" >0.0040s</td>
    </tr>
    <tr>
		<td>Naive Bayes</td>
		<td style="text-align: right" class="danger">40.86%</td>
		<td style="text-align: right" >0.1988s</td>
		<td style="text-align: right" >0.0224s</td>
    </tr>
    <tr>
		<td>QDA</td>
		<td style="text-align: right" class="danger">44.25%</td>
		<td style="text-align: right" >1.9541s</td>
		<td style="text-align: right" >0.0483s</td>
    </tr>
    <tr>
		<td>RBM 100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >31.7963s</td>
		<td style="text-align: right" >0.0137s</td>
    </tr>
    <tr>
		<td>RBM 100, n_iter=20</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >62.9672s</td>
		<td style="text-align: right" >0.0137s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=1</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >218.7416s</td>
		<td style="text-align: right" >0.0232s</td>
    </tr>
    <tr>
		<td>RBM 200, n_iter=40, LR=0.01, Reg: C=10000</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >218.5600s</td>
		<td style="text-align: right" >0.0235s</td>
    </tr>
    <tr>
		<td>RBM 256</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >75.7329s</td>
		<td style="text-align: right" >0.0276s</td>
    </tr>
    <tr>
		<td>RBM 512, n_iter=100</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >146.0882s</td>
		<td style="text-align: right" >0.0520s</td>
    </tr>
    <tr>
		<td>Random Forest</td>
		<td style="text-align: right" class="danger">47.01%</td>
		<td style="text-align: right" >16.1065s</td>
		<td style="text-align: right" >3.1128s</td>
    </tr>
    <tr>
		<td>Random Forest 2</td>
		<td style="text-align: right" class="danger">46.12%</td>
		<td style="text-align: right" >0.1387s</td>
		<td style="text-align: right" >3.1423s</td>
    </tr>
    <tr>
		<td>SVM, adj.</td>
		<td style="text-align: right" class="danger">48.28%</td>
		<td style="text-align: right" >1661.0278s</td>
		<td style="text-align: right" class="danger">56.2758s</td>
    </tr>
    <tr>
		<td>SVM, linear</td>
		<td style="text-align: right" class="danger">47.83%</td>
		<td style="text-align: right" >1191.7719s</td>
		<td style="text-align: right" class="danger">47.0907s</td>
    </tr>
    <tr>
		<td>k nn</td>
		<td style="text-align: right" class="danger">39.04%</td>
		<td style="text-align: right" >1.2344s</td>
		<td style="text-align: right" class="danger">65.0951s</td>
    </tr>
</tbody>
</table>

