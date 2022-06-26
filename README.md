# ***小样本下材料领域的命名实体识别***
*北京大数据技能大赛*

取名字好难



------

## **摘要**

随着新出版的科学文献不断增加，依赖专家进行数据提取是费时和劳动密集型的。因此，能够自动从文献中提取重要信息的命名实体识别 (NER)技术逐渐获得广泛关注。然而，材料领域中相关实体名称组成复杂、结构嵌
套，且缺乏大规模人工标注语料，为抽取统一、完整、准确的实体命名带来了困难。本文基于初赛数据集的分析和广泛的文献调研，提出了包含上下文推理的标注规则与高效的标注方法，目前构建了约 150 份与主题相关的更加适合小
样本下材料领域的实体命名数据集，结合迁移学习与负采样等技术设计了多种模型，包括 Word2Vec+LSTM+CRF 模型、Word2Vec+LSTM+CRF 模型以及 BERT 预训练模型+CRF。本文以预测大标签为目标，数据结果表明：三种模型的平均 F1 值在 0.88 以上，整体学习效果较好。其中，BERT+CRF模型表现最好，模型的 F1 值达到 0.8971。在引入负采样技术后，能够进一步提高小样本下少数标签的识别率和 F1 值。此外，针对小样本数据集的特
点，本工作初步探索了半监督学习的可行性。预计在获得更多数据后，模型的性能将表现更佳，为基于数据驱动的机器学习在材料领域的应用奠定基石，以加速新材料的设计与发现。

#### 关键词： 小样本 材料领域 BERT CRF 负采样 半监督

​开源项目地址为https://github.com/Xzaohui/chemical_ner

------

## **问题描述**

​从材料领域科学文献中准确抽取相关命名实体是对该领域知识进行深层次分析的基础，对材料属性预测、催化方案生成以及新材料发现等方面具有重大意义。然而，材料领域中相关实体名称组成复杂、结构嵌套，且缺乏大规模人工标注语料，为抽取统一、完整、准确的领域实体带来了困难。

​要求基于从材料领域科技文献数据库中抽取的专家标记的语料库，设计基于深度学习（包括但不限于CRF、LSTM、BiLSTM、Transformer等）及预训练模型的智能模型与方法，完成材料领域的催化原料、催化反应、催化生成物、法拉第效率属性等命名实体的识别任务，助力材料领域创新。



------

## **解决方案**

​从数据和模型两个方面切入，我们构建了高质量的数据集与多种神经网络模型，力求达到任务要求。

​在数据方面，为解决数据集的标注规则不统一和标注密度过低的问题，我们基于初赛数据集的分析与整理，提出了更适合小样本下材料领域实体命名的标注方案，并由相关专业的参赛队员人工标注约500份的高质量数据集。

​在模型方面，我们采用谷歌公司最新开发的与化学相关的*BERT*模型作为预训练模型加上条件随机场CRF作为标签序列预测，同时以*LSTM+CRF*和*LSTM*+*CRF*模型作为baseline和最新的模型作对比。

​实验结果表明，目前BERT+CRF+负采样模型表现最佳，整体F1值达到0.8979，关键数据MRPF标签平均F1值达到0.3543。从损失函数曲线分析可知，进一步扩大高质量的数据集规模，模型的表现效果可以进一步提升。这充分表明我们整体设计方案的有效性。

## **创新点**

1. 基于初赛数据集的分析与整理，提出了新的更适合小样本领域材料实体命名的标注规则：建立统一、逻辑合理且包含上下文推理的标注规则

   1. 设立了额外的大标签进行多任务目标实体预测，以提高模型对主任务的预测效果
   2. 广泛采用字典方式进行自动标注，辅助和加快繁琐的人工标注过程。

2. 使用了当下由谷歌公司开发的最新的*BERT*预训练模型，采用迁移学习与fine-tune微调的方法对模型进行训练，对小样本数据集更为友好。
3. 针对专业领域的标注数据集较小、相对复杂的标注规则以及大量的非实体标签（O），我们设计了Self-training半监督学习和负采样技术对训练进行了优化，大大增加了对于极少出现的标签识别的准确率。

## **初赛数据集分析**

1. 数据集的样本总数分别为 243 份，由化学领域的小组成员和大赛相关人员一致确认数据集的主题为“基于*Cu*催化*CO*2还原”。
2. 数据集的大标签一共有44种，包括：材料种类，调控因素，产物种类和法拉第效率，小标签共计32种，大小标签出现的频数统计如下所示：
   1. 大标签统计情况，包括材料种类（Materials types: M）,调控因素(Regulatory factors: R) ,产品种类(Product categories, P) 以及 Faradaic efficiency( F ) ：
   ![img](https://docimg7.docs.qq.com/image/Y2X5KMR3N8F92Ndgap_AaQ.png?w=640&h=480)
   2. 小标签统计情况如下：
​                 ![img](https://docimg1.docs.qq.com/image/ZbtrLmUPNNhsLkHmxtWqkQ.png?w=640&h=480)        



3. 数据集的data内容中，一共出现6161种单词，出现的高频词（展示前20个）统计结果如下：

​                 ![img](https://docimg7.docs.qq.com/image/MTuxgteLuSeQjVqTorsbVQ.png?w=640&h=480)        

4. 数据集中每个样本的标注次数平均为3.5次，而数据集的data标签内的单词总数分布图如下所示：

​                 ![img](https://docimg6.docs.qq.com/image/3WYuPrzsgTorE1PU4sAKxg.png?w=603&h=446)        

​	每个样本的单词总数分布主要从500到2500之间，而每个样本的平均标注次数为3.5次。表明标注的密度过低，数据中大部分单词的标注为 O ，不利于模型训练。

5. 每个标签标注的字符串存在规则不统一的情况，以Faradaic efficiency为例，有样本标记原文的长达103个字符（49.7% and a high current density of 28.5 ma cm(-2) at -1.19 v vs. a reversible hydrogen electrode (rhe)），而有的样本则几个字符（90.24 %或17.9)。

6. 标签中存在少量的明显问题标注，如82.3%应该标记为Faradaic efficiency，但实际却被标记为HCOOH

​	仅使用初赛的243份数据集，训练的*NLP*模型难以收敛，无法对模型做进一步的改进；同时，初赛数据集存在标注规则前后不一致的问题。因此，需要人工构建新的数据集并建立统一的标注规则。

7. 初赛数据集漏标严重，重复出现的应该被标注的单词以及词组并没有被标注。

## **数据收集**

​	从web of science网站，以"metal" OR "cu" OR "copper" (主题) and catalyzed reduction (主题) and "carbon dioxide" OR "CO2" (主题) 为检索条件，检索与初赛数据集主题十分相关的新样本数据，作为我们下一步人工标注的数据来源，共计1661份（截止5.17日）。新数据集中仅与初赛数据集一共存在17篇重复。

​	另外，从web of science网站，以cu (主题) and copper (主题) or co2 (主题) or carbon (主题) and reduction (主题) and cata (主题)为检索条件，检索到1,178,4251,178,425份结果。依据相关性排序，下载了共12,869份的样本数据集，作为模型词向量的来源。

## **新数据集与初赛数据集比较分析**

​	初赛数据集的243份数据中，出现了6161种单词。除了介词、冠词，以及已经是标签的除外，频数最高的分别是('CO2', 844) ('reduction', 463) ('electrochemical', 270) ('selectivity', 266) ('catalysts', 208) ('catalyst', 198)('surface', 190)('reaction', 164)；

​	新数据集的1616份数据中，出现了49926种单词。除了介词、冠词，以及已经是标签的除外，频数最高的分别是：('CO2', 2735)('reduction', 1664)('reaction', 1315)('catalytic', 895)('catalysts', 878)('catalyst', 818)('surface', 606)('catalyzed', 508)('reactions', 466)。

## **标签说明**
二氧化碳是一种主要的温室气体，容易导致全球气候急速变暖，一般可
以通过植物的光合作用减少二氧化碳在空气中的含量，但由于效率较低时
间较慢，保护环境迫在眉睫，故可以将二氧化碳进行进一步工业化利用转化
为许多下游碳基化合物，一来助力国家双碳战略目标，二来发展产业工业
化。在其中电化学转化是近年来较受关注的一个领域，在电化学法还原中，
铜金属做催化剂还原二氧化碳因其还原效果好、活性高受到广泛关注。催化
剂金属的种类即为研究问题的目标材料种类，调控因素为催化剂金属的形
貌特征，产物种类是生成物的不同种类，在电化学催化反应中法拉第电磁感
应效率是表征催化剂活性的重要标准。故设置四类大标签：材料种类 M、调
控因素 R、产物种类 P 和法拉第电磁感应效率 F。
材料种类就是催化剂的种类，有单金属、金属氧化物、金属硫化物、双
金属、双金属合金、MOF 金属-有机框架材料等等。调控因素主要是指催化
剂材料的形态结构的影响因素，目标数据集中结构控制、表面因素等都是主
要影响因素，通过影响材料的结构决定材料的物理化学催化性能。产物种类
12
分为一碳产物、二碳产物和多碳产物，以及大量碳氢、碳氢氧化物，均为附
加值较高的二氧化碳转化下游产品。

## **标注规则**

​由于初赛数据集存在许多不利于模型训练的问题，诸如：每个标签标注的字符串存在逻辑不统一，无法依据上下文推理正确的以及标注密度过低等问题。因此，建立统一、逻辑合理且包含上下文推理的标注规则十分必要。

​首先，由于小样本的数量量过低，仅有数百份，每份的平均标记次数为3.5，而小标签共计32种。因此，预测每个小标签几乎是难以做到事情。故出于保证模型有效的目标下，以预测四种大标签为目标。

​其次，由于一篇文章可能出现多种催化原料的相互比较或是多种催化产物，但文章往往强调某一种催化原料或催化产物是最可取方案，而其它物质是作为较差的对比项出现，即模型需要根据上下文推理或识别出正确的物质，如Cu在某一篇文章中可能是最佳的方案，但在另一篇文章中可能不是。因此，新标注规则不仅需要对样本中正确的物质或因素进行标记，也需要对样本中出现的“次要物质”或“次要因素”也进行标注。

​最后，由于标注密度过低，非标注字词占比过高，即样本的标签分布极不平衡，这对模型训练存在影响很大。因此，需要补充额外的大标签。经过新数据集与初赛数据集比较分析后，结果表明两数据集中以CO2 ,reduction ,（catalytic/catalysts/catalyst/catalyzed）以及(electrochemical, electrodes, electrode, electroreduction, electrocatalysts, electrocatalytic 这四类词出现频次最高，且与的样本主题（“基于Cu催化CO2还原”）存在较强的相关性，因此，选取这四类词作为额外的大标签，缓解标注密度过低的问题，同时能够辅助模型进行上下文推理，提高模型最终的预测效果。

​进一步的统计分析表明，这四类额外大标签几乎可以使用字典进行自动标注。相似的，初赛数据集的分析结果表明32种小标签中，特别是部分的催化原料和产物种类，可以用穷举法进行完全枚举。因此，在实际的新样本标注过程中，首先使用初赛数据集的标注结果与基于词频统计的结果生成字典，采用字典对新数据集进行自动标注，再由人工进行核对与修改。

## **数据整理**

​目标的大标签分别是材料种类（Materials types: M）,调控因素(Regulatory factors: R) ,产品种类(Product categories, P) , Faradaic efficiency( F ) 。额外的大标签为二氧化碳（'CO2'：'CO2'，'carbon dioxide']），还原（'RE'：'reduction')，电相关词('ELE':'electrochemical','electrodes','electrode','electroreduction',etc. )以及催化('CA': 'catalytic','catalysts','catalyst','catalyzed' )。

# **项目细节**

## **任务介绍**

​本次项目任务小样本下材料领域实体识别是在自然语言处理领域很常见的序列标注任务，序列标注的的输入是一个序列，他的输出也是一个序列，他的典型的例子就是词性标注任务（pos tagging）和命名实体识别任务（ner）。对于序列标注任务这种时序问题最主要使用的模型就是循环神经网络RNN，而RNN中的LSTM长短时记忆网络以及它的变种BiLSTM双向LSTM是对序列长时依赖更有效的模型，同时最近几年由谷歌公司开发的BERT模型也在此任务方向上取得了非常不错的效果。


## **模型选择**

​本次项目目前选择的模型为BERT+CRF，传统baseline模型选择为LSTM+CRF和BiLSTM+CRF。后续会根据项目进度实现更新的研究成果和模型。

## **模型介绍**

### **LSTM&BiLSMT**

​LSTM的全称是Long Short Term Memory，顾名思义，它具有记忆长短期信息的能力的神经网络。LSTM首先在1997年由Hochreiter & Schmidhuber提出，由于深度学习在2012年的兴起，LSTM又经过了若干代大牛的发展，由此便形成了比较系统且完整的LSTM框架，并且在很多领域得到了广泛的应用。

​LSTM提出的动机是为了解决上面我们提到的长期依赖问题，较长的序列在传统RNN中循环输入输出，较早的序列信息对后续序列的影响较小，而LSTM引入了门（gate）机制用于控制特征的流通和损失，包含记忆Cell、遗忘门、输入门、输出门等结构。

​                 ![img](https://docimg5.docs.qq.com/image/L8ondlHis50TF3j3kap99g.jpeg?w=1440&h=541)        

​同时我们注意到无论是传统的RNN还是LSTM，都是从前往后传递信息，这在很多任务中都有局限性，比如词性标注任务，一个词的词性不止和前面的词有关还和后面的词有关。为了解决该问题，设计出前向和方向的两条LSTM网络，被称为双向LSTM，也叫BiLSTM。其思想是将同一个输入序列分别接入向前和先后的两个LSTM中，然后将两个网络的隐含层连在一起，共同接入到输出层进行预测。

​                 ![img](https://docimg6.docs.qq.com/image/EO7LStNZ-8WBnMsel5bBSg.png?w=640&h=421)        

#### **模型的输入与输出**

​LSTM和BiLSTM模型的输入结构为X=(Batch_size,Max_sentence_len,Embedding_dim)，由于样本量较小，我们选择较小的Batch size=1和较小的learning rate，经过统计样本数据，我们将Max_sentence_len定为600，Embedding层我们使用word2vec作为预训练模型，Embedding_dim目前定为256。

​我们需要的输出为Y=(Batch_size,Max_sentence_len,Len_of_label)，而Y(i,j,k)=p的含义是第i个Batch下第j个单词的标签为k的概率。我们有27种标签，而LSTM的隐藏层hidden dim=512，因此需要大小为linear=(512,27)的全连接层，而BiLSTM的隐藏层hidden dim=64，因此需要大小为linear=(128,27)的全连接层。

### **BERT**

​BERT是2018年10月由Google AI研究院提出的一种预训练模型。BERT的全称是Bidirectional Encoder Representation from Transformers，可以知道BERT模型实际上是使用transformer作为算法的主要框架，双向的Transformers模型的Encoder部分，是一种典型的双向编码模型。它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的masked language model（MLM）和next sentence prediction的多任务训练目标，是一个自监督的过程，不需要数据的标注。使用tpu这种强大的机器训练了大规模的语料，是NLP的很多任务达到了全新的高度。

​                 ![img](https://docimg3.docs.qq.com/image/jyu6HY5NJRoNGF3mETZomg.png?w=887&h=666)        

​BERT模型十分巨大，模型分为base和large两个版本，base版本由12层Transformers模型的Encoder部分组成，有768个隐藏层参数，总参数量有1.1亿个，而large版本由24层Transformers模型的Encoder部分组成，有1024个隐藏层参数，总参数达到了3.4亿个。

​BERT如此巨大的模型也需要庞大的数据量和计算资源，因此BERT模型一般是由大机构或研究所预训练完成后上传至HuggingFace作为开源预训练模型使用，我们可以比较方便地在自己的数据集上进行fine-tune微调。

​BERT模型的每个目标词是直接与句子中所有词分别计算相关度(attention)的，所以解决了传统的RNN模型中长距离依赖的问题。通过attention，可以将两个距离较远的词之间的距离拉近为1直接计算词的相关度，而传统的RNN模型如LSTM&BiLSTM中，随着距离的增加，词之间的相关度会被削弱。



#### **模型的输入与输出**

​BERT模型的输入与LSTM&BiLSTM模型不同，输入的向量是由三种不同的embedding组合而成，分别是：

1. wordpiece embedding：单词本身的向量表示。WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。
2. position embedding：将单词的位置信息编码成特征向量。因为我们的网络结构没有RNN 或者LSTM，因此我们无法得到序列的位置信息，所以需要构建一个position embedding。构建position embedding有两种方法：BERT是初始化一个position embedding，然后通过训练将其学出来；而Transformer是通过制定规则来构建一个position embedding
3. segment embedding：用于区分两个句子的向量表示。这个在问答等非对称句子中是用区别的。

​虽然输入较LSTM&BiLSTM复杂，但BERT模型有其自带的tokenizer工具，因此在输入方面比较简单，只需要将句子输入tokenizer中，就能将其分成input_ids、attention_mask和offset_mapping，后续只需要将offset_mapping转化为我们的标签信息即可将这三项输入模型中。

​                 ![img](https://docimg4.docs.qq.com/image/hdrapGYuViSFoNkN20mZFg.png?w=984&h=321)        

​BERT模型的输出与传统LSTM&BiLSTM模型的输出相同，为Y=(Batch_size,Max_sentence_len,Len_of_label)，而Y(i,j,k)=p的含义是第i个Batch下第j个单词的标签为k的概率。



### **CRF模型**

条件随机场即CRF模型可以在在新的sample（观测序列 ）上找出一条概率最大最可能的隐状态序列 。

​                 ![img](https://docimg3.docs.qq.com/image/YG32K0StyDomPqfbp3QARg.png?w=415&h=197)        

​CRF与HMM隐马尔可夫模型结构类似，但CRF是无向图，HMM为有向图，CRF是判别式模型，HMM为生成式模型，CRF在如今的NLP领域应用更加广泛。

​单独的LSTM&BiLSTM抑或BERT也能够通过输出各个token的各个label概率来预测标签的序列，但是它们都不能学习到标签之间的条件转移，而CRF是全局范围内统计归一化的条件状态转移概率矩阵，再预测出一条指定的sample的每个token的label，因为CRF的特征函数的存在就是为了对given序列观察学习各种特征（n-gram，窗口），这些特征就是在限定窗口size下的各种词之间的关系，然后一般都会学到这样的一条规律（特征），因此加入CRF模型会大大提升整体的预测准确率和合理性。

CRF的建模公式如下：

​                 ![img](https://docimg1.docs.qq.com/image/ViTT4D-ElHswj-WxtJJCTg.svg?w=675&h=45&type=image%2Fsvg%2Bxml)        

​其分子为路径分数的指数，分母为归一化的整体路径分数的指数之和，即可以理解为：

​                 ![img](https://docimg9.docs.qq.com/image/gyabuzYyPz01KeDiCK-_Mw.svg?w=568&h=72&type=image%2Fsvg%2Bxml)        

​我们当然是想让正确路径的概率越接近于1越好，因此可以令loss=-log(p(l|s)作为损失函数训练模型。

### **Word2Vec**
word2vec是Google研究团队里的Tomas Mikolov等人于2013年的《Distributed Representations ofWords and Phrases and their Compositionality》以及后续的《Efficient Estimation of Word Representations in Vector Space》两篇文章中提出的一种高效训练词向量的模型，基本出发点和Distributed representation类似:上下文相似的两个词，它们的词向量也应该相似，比如香蕉和梨在句子中可能经常出现在相同的上下文中，因此这两个词的表示向量应该就比较相似。

我们选择由6000余篇相关领域的论文数据训练出Word2Vec预训练模型代替随机初始化的Embedding层，使得我们的词向量包含更大的信息量帮助我们的模型学习更深层次的信息。

## **其他训练算法**

### **Self-training&Co-training**

半监督学习是一种介于监督式学习和无监督学习之间的学习范式，我们都知道，在监督式学习中，样本的类别标签都是已知的，学习的目的找到样本的特征与类别标签之间的联系。一般来讲训练样本的数量越多，训练得到的分类器的分类精度也会越高。但是在很多现实问题当中，一方面由于人工标记样本的成本十分高昂，导致了有标签的样本十分稀少。而另一方面，无标签的样本很容易被收集到，其数量往往是有标签样本的上百倍。半监督学习就是要利用大量的无标签样本和少量的有标签样本来训练分类器，解决有标签样本不足这个难题。

​将初始的有标签数据集作为初始的训练集，根据训练集训练得到一个初始分类器。利用初始分类器对无标签数据集中的样本进行分类，选出最有把握的样本，如本次项目实验中可以选择以路径分数为判断依据。而后将选择出的样本加入到有标签数据集中对模型进行训练，随后根据新的训练集训练新的分类器，重复步骤2到5直到满足停止条件（例如所有无标签样本都被标记完了）最后得到的分类器就是最终的分类器。

​但由于试验结果不稳定，本次报告并没有加入此项训练方法，后续研读更多最新论文后可以继续改进。

### **负采样**

​NER数据会存在大量漏标，实体标注应该算是NLP中比较复杂的，需要专业标注知识、需要统一标注规范。NER数据中存在大量实体，标注员想要把所有实体都标注出来是不现实的，因此数据存在漏标也不可避免。特别是在专业领域小样本下的命名实体识别，在本身数据量较小的情况下更容易收到此类噪声的影响。把未标注的实体当作“负样本”就是一种噪声，因为漏标的实体不应当做标签为O的负样本来看待。

​未标注实体问题会导致NER指标下降。主要有2个原因：一是实体标注量减少；二是把未标注实体当作负样本。其中第二个原因起主要作用。因此需要对所有非实体片段进行负采样（下采样）。这也很好理解：所有非实体片段中，有一部分可能是真正的、但未标注的实体（也就是未标注实体），但我们把能把它们都当作“负样本”看待，因此需要对所有非实体片段进行负采样。我们可以简单的以一个小概率（5%）将O随机标注为M、R、P、F四个标签中的一个，类似于噪声的数据，同时保证在整个数据集上的概率归一化以保证不
会产生过大的偏差，后续对负采样的优化和改进仍在研究当中。





## **实验结果**

项目实验平台为python 3.6.13 torch 1.10.2 cuda 11.6

项目实验目前标签识别的整体测试值如下：

| 模型                     | F1 score |
| ------------------------ | -------- |
| W2V+LSTM+CRF             | 0.8793   |
| W2V+BiLSTM+CRF           | 0.8958   |
| BERT+CRF                 | 0.8971   |
| W2V+LSTM+CRF+负采样      | 0.8897   |
| W2V+BiLSTM+CRF+负采样    | 0.8847   |
| BERT+CRF+负采样          | 0.8979   |
| BERT+CRF+负采样+official | 0.9803   |


由表 1 可以看出各个模型学习效率较好，平均 F1 值可达到 0.88 以上，
但组委会数据的训练结果 F1 值达到 0.98 以上十分反常，猜测是由于非实体标签 O 过多导致的部分过拟合现象，大量标签被标注为 O 使得 F1 值虚高。


项目实验目前关键标签（M、R、P、F）识别的测试值F1如下：

| 模型                     | B-M    | I-M    | B-R    | I-R    | B-P    | I-P    | B-F    | I-F    | average |
| :----------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| W2V+LSTM+CRF             | 0.4667 | 0.0606 | 0.1034 | 0.2000 | 0.5882 | 0.2500 | Nan    | Nan    | 0.2086  |
| W2V+BiLSTM+CRF           | 0.6218 | 0.1270 | 0.0364 | Nan    | 0.6452 | 0.2609 | Nan    | Nan    | 0.2114  |
| BERT+CRF                 | 0.6667 | 0.2952 | 0.2295 | Nan    | 0.6377 | 0.2449 | Nan    | Nan    | 0.2588  |
| W2V+LSTM+CRF+负采样      | 0.6087 | 0.2192 | 0.1194 | 0.1463 | 0.6452 | 0.2609 | Nan    | Nan    | 0.2500  |
| W2V+BiLSTM+CRF+负采样    | 0.4615 | 0.0869 | 0.0689 | 0.0357 | 0.6557 | 0.3333 | 0.2667 | 0.3636 | 0.2840  |
| BERT+CRF+负采样          | 0.6949 | 0.2857 | 0.1852 | 0.1322 | 0.6769 | 0.2449 | 0.2789 | 0.3356 | 0.3543  |
| BERT+CRF+负采样+official | 0.5882 | 0.3636 | Nan    | Nan    | 0.7500 | Nan    | Nan    | Nan    | 0.2127  |


由表 2 可以看出试验结果基本符合最初设想，W2V+LSTM+CRF 到
W2V+BiLSTM+CRF 再到 BERT+CRF 模型复杂程度增加，关键数据的
识别率依次提高，同时在整体 F1 值变化不大的情况下，增加了负采样的模
型在小样本下对少数标签的识别率和 F1 值，同时也验证了上一个对组委会
数据训练结果的猜想，其 F1 值虚高，而对关键数据的识别率和 F1 值较低。

## 总结分析

​横向对比各个模型，BERT+CRF模型的F1值明显高于另外两种传统W2V+LSTM+CRF/W2V+BiLSTM+CRF模型，并且负采样技术可以增加标签的识别率，特别是对极少出现的标签识别率有质的提高，如R、F标签。

​纵向来分析各个模型的数据可以得出目前三种模型的平均F1值在0.88以上，模型的学习效率较好。但大多是由于无关且变化较少标签如ELE、CA、CO2等以及非实体标签O识别率较高的影响，而关键标签M、R、P、F的识别率由于数据量过于稀少，并不十分理想。特别是由于官方数据标签稀少，如果只看整体的F1值官方数据非常漂亮，但是其关键数据识别率较我们的数据训练结果差很多，因此我们重新对数据进行处理是非常有必要的。

​后续项目应继续完善数据标注，继续增加数据集，同时可以在半监督学习、不完全实体标注问题、机器阅读理解MRC模型等方向进一步研究学习，提升算法效率。
