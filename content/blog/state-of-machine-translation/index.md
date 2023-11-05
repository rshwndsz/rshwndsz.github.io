---
title: "A survey on Machine Translation"
bibliography: bibliography.bib
link-citations: true
notes-after-punctuation: true
description: A concise look at the history, current challenges and future directions of research in machine translation.
---

## Introduction

### What is machine translation? 

Machine Translation [MT] refers to computerised systems that translate
natural languages without any human assistance. The goal of
sentence-level MT is to find the most probable target sentence
$\bf \hat y$ given a source sentence $\bf x$ such that the target
conveys the same meaning as the source sentence. Mathematically, this
can be expressed as:
$$
\bf{\hat y} = \underset{\bf y}{arg\,\max}\; P_{\theta} (\bf y \mid \bf x) 
$$
Modelling the conditional probability $\bf P(\bf y \mid \bf x)$ with
learnable parameters $\theta$ is done using various MT models and
techniques ranging from rule-based and statistical models to neural
machine translation [NMT] models. Most existing NMT models are
auto-regressive, i.e. they define a probability distribution over target
sentences $\bf P(\bf y \mid \bf x)$ by factorising it into individual
conditionals as 
$$
    \bf P(\bf y \mid \bf x)  = \prod_{i = 1}^{N} \bf P(y_i \mid y_1, \ldots, y_{i - 1}, \bf x) \tag{1}
$$
where $y_i$ is the current target word and $y_1, \ldots, y_{i - 1}$ are
previously generated words. Once $\bf P(\bf y \mid \bf x)$ is learned by
a translation model, a source sentence is translated by searching for
the sentence that maximises the conditional probability
[@sutskever2014sequence; @maruf2021survey].

### Why is machine translation a problem worth solving?

The ability to communicate effectively is essential for human
interaction and development, particularly in fields such as science,
medicine, and technology, where collaboration between people from
different countries is essential for progress. However, language
barriers can often hinder communication, especially in a globalised
world where people from different cultures and countries interact
frequently. A study by [@lee2020impact] exploring the impact of MT on
English-as-a-Foreign-Language students' writing skills in Korea showed
that the group with access to machine translation tools produced essays
with significantly higher accuracy and complexity scores than the
control group. Improvements in data-collection and model training
allowed Google to add 24 new languages to Google Translate in one go,
benefitting under-represented speaker populations in Africa and South
Asia [@51503]. In a recent paper, [@khoong2022research] argue that MT
has the potential to improve communication between healthcare providers
and non-native speakers, leading to better and more equitable healthcare
outcomes. However, the field still faces many challenges, from needing
large-scale datasets for low-resource languages to adapting systems to
specialised domains.

## Brief Overview of Approaches

Machine translation has come a long way since its formal inception in
the late 1940s and its first public demonstration by the Georgetown-IBM
research group in 1954. An overview of the ancient arts of rule-based
and statistical MT systems can be found in [@hutchins1997first]. This
section focuses on different neural-network-based machine translation
systems and specifically attention-based approaches, which are also
expanded upon in [a later section](#large-language-models). Neural models have become the *de-facto*
standard and are consistently approaching human-level performance in
various settings [@hassan2018achieving]. NMTs are also being widely
adopted in industry and have seen deployments in many large production
systems [@schmidt2018move; @51503]

### The Encoder-Decoder Framework 

The encoder-decoder structure, first proposed by
[@neco1997asynchronous], is the current de-facto standard for NMT
models. These systems are characterised by an encoder network which
computes a latent representation of the source sentence, followed by a
decoder network which generates the translated sentence from that
representation. Different encoder-decoder architectures model the
individual conditional $\bf P(y_i \mid y_1, \ldots, y_{i - 1}, \bf x)$
from Eq. 1 differently. Recurrent neural networks \[RNNs\]
were first introduced to model the distribution as a function of the
current word given previously generated words along with some hidden
state and fixed-length representation of the input.

### Before Transformers

[@kalchbrenner-blunsom-2013-recurrent-convolutional] were among the
first to present a standalone NMT system without components from
statistical MT \[SMT\]. They demonstrated using a convolutional neural
network \[CNN\] based encoder to model sentence pairs to capture
syntactic and lexical features of the input sentences. Following this
line of research, [@sutskever2014sequence] and [@cho-etal-2014-learning]
explored the use of stacked LSTMs and GRUs in the encoder, respectively,
to generate a fixed-length encoding of the source sequence. However,
fixed-length source encodings have been shown to lead to poor
translations for long input sentences, as reported by
[@cho-etal-2014-properties]. To address the performance bottleneck of
fixed encodings, [@DBLP:journals/corr/BahdanauCB14] proposed the
attention mechanism. This approach allows the model to attend to
specific parts of the input sequence while generating the output,
negating the need for fixed input representations.

### The Transformer era 

Sequential models provided a significant increase in performance
compared to traditional SMT techniques. However, their use in
large-scale machine translation was and continues to be limited by the
challenge of parallelising training examples, which creates a bottleneck
in processing longer sentences. [@vaswani2017attention] proposed the
Transformer architecture to replace traditional recurrent and
convolutional neural network layers. The authors presented an
improvement over the vanilla attention mechanism
[@DBLP:journals/corr/BahdanauCB14] with 'self-attention', which allows
the Transformer to learn global dependencies between the words in the
sequence, enabling the generation of more informative and
context-sensitive word embeddings. These embeddings, called
'contextualised embeddings' because they are generated by considering
the entire input sequence, have been shown to significantly outperform
traditional fixed source encodings and improve the model's performance
on various natural language processing tasks, including machine
translation. The paper also described another novel mechanism called
'multi-head attention', which stacks multiple self-attention 'heads' in
parallel to enable the model to attend to different positions in the
input sequence simultaneously, improving the quality of the learned
representations while also making the model parallelisable.

Every aspect of the vanilla Transformer has been improved and modified
in various ways to improve its performance, from the attention mechanism
[@8894858; @guo2020multi], and positional encodings
[@devlin-etal-2019-bert; @dai-etal-2019-transformer] to the activation
functions of the feed-forward networks
[@DBLP:conf/iclr/RamachandranZL18; @chen2020generative].
[@devlin-etal-2019-bert] introduced a novel language representation
model - Bidirectional Encoder Representations from Transformers \[BERT\]
that pre-trains deep bidirectional representations from unlabeled text
by jointly conditioning on both left and right contexts in all layers,
which allows it to capture a deeper understanding of language. The
authors also proposed a novel pre-training objective called 'Masked
Language Modeling', which involves randomly masking some input tokens
and training the model to predict the masked tokens. BERT achieved new
state-of-the-art results on 11 NLP tasks, including machine translation
and has become the basis for many subsequent advances in the field
[@lanalbert].

## Key Challenges and Current Work

### Datasets 

One of the main challenges in machine translation is the availability of
large, high-quality datasets for training and evaluating models. Over
the years, several datasets have been developed specifically for machine
translation research. The Workshop on Machine Translation \[WMT\] has
been running an annual evaluation campaign since 2006
[@zerva-etal-2022-findings; @specia2020findings; @koehn2018findings; @buck2016findings],
which includes a shared task for machine translation. The datasets used
in this task are typically parallel corpora of news articles covering a
range of languages, including English, German, French, and Chinese. The
International Workshop on Spoken Language Translation \[IWSLT\] is a
yearly workshop focusing on spoken language translation
[@antonios2022findings; @ansari2020findings; @iwslt-2018-international; @cettolo2016iwslt].
The datasets used in this workshop include audio recordings of speeches,
as well as transcripts and translations in various languages. In recent
years, datasets have only gotten more extensive and diverse, enabling
more complex translation tasks and models. XTREME [@hu2020xtreme] is a
benchmark dataset for evaluating the cross-lingual generalisation
capabilities of pre-trained multilingual models covering 40
typologically diverse languages and 9 tasks, including machine
translation. Flores-101 [@goyal2022flores] is a benchmark dataset for
low-resource machine translation, which consists of parallel sentences
in 101 languages, making it one of the most extensive multilingual
machine translation datasets available.

### Evaluation 

Numerous evaluation metrics have been proposed to evaluate the quality
of the generated translations. The most popular of them, BLEU, short for
Bilingual Evaluation Understudy, has been the *de-facto* standard for
evaluating translation outputs since it was first proposed by
[@papineni2002bleu]. The core idea of BLEU is to aggregate the count of
words and phrases that overlap between machine and reference
translation. BLEU scores range from 0 to 1, where 1 means a perfect
translation. However, using BLEU directly is suboptimal because it
relies on $n$-gram overlap, which is heavily dependent on the specific
tokenisation used. Tokenising aggressively can artificially raise the
score and make comparing results across different studies difficult.
SacreBLEU [@post-2018-call] addresses this challenge by providing a
hassle-free computation of shareable, comparable and reproducible BLEU
scores. Human evaluation, however, is still considered the gold standard
in this field as it takes into account the nuances of language that can
be difficult for machines to capture. Human evaluators can assess not
only the translation's accuracy but also the output's fluency and
naturalness. In addition, human evaluation can provide valuable insights
into the text's cultural context, which can be crucial for producing
high-quality translations. MT evaluation is an active research area and
was also the WMT shared task for 2022 [@zerva-etal-2022-findings], where
participants had to predict the quality of generated translations
without access to references.

### Low-resource languages 

The vast majority of improvements made in machine translation in the
last decades have been for high-resource languages, i.e. the languages
that have large quantities of training data available digitally
[@fan2021beyond]. High-resource languages like English, French and
Japanese rarely have dataset size concerns. For instance, the
English--French corpus used by [@cho-etal-2014-properties] as early as
2014 contained 348 million parallel sentences. However, low-resource
languages have not received enough attention from the NLP community
despite being widely spoken around the world due to a multitude of
reasons: lack of state investments, no codified research norms, lax
organisational priorities, Western-centrism and logistical challenges in
procuring training data to name a few [@costa2022no]. While NMT systems
have demonstrated remarkable performance in high-resource data
scenarios, research has indicated that these models exhibit low data
efficiency and perform worse than unsupervised methods or phrase-based
statistical machine translation in low-resource conditions
[@koehn-knowles-2017-six]. However, recent research has demonstrated
that NMT is suitable in low-data settings but is very sensitive to
hyperparameters such as vocabulary size, word dropout, and others
[@sennrich-zhang-2019-revisiting]. A recent initiative towards
rectifying the lack of resources for low-resource languages is the
FLORES-101 benchmark by [@goyal2022flores], which consists of the same
set of English sentences translated into 100 other languages. However,
it has the limitation that for non-English pairs, the two sides are
\"translationese\" and not mutual translations of each other.

### Domain[^1] adaptation 

NMT systems struggle in scenarios where words have different
translations, and their meaning is expressed in different styles in
different domains. For example, a model trained exclusively on law
reports is unlikely to perform well in clinical medicine
[@zhang-etal-2019-curriculum]. It has been shown that NMT systems drop
in performance when training and test domains do not match and when
in-domain training data is scarce [@koehn-knowles-2017-six]. This is of
particular concern when machine translation is used for information
summarising - users are likely to be misled by hallucinated content in
the generated translation. A naive solution is to tailor the NMT model
to every specific domain. In addition to being a highly impractical
approach, high-quality parallel data only exists for some domains, and
often, large amounts of training data are only available out of domain.
[@luong-etal-2015-effective] demonstrated that a pre-trained system can
be repurposed to translate new domains more quickly than training a new
model and often performs better on the new domain.

### Decoding 

The task of finding the most likely translation $\bf \hat y$ for a given
source sentence $\bf x$ is known as the *decoding* problem. Decoding in
MT is a challenging problem as the search space grows exponentially with
sequence length making a complete enumeration of the search space
impossible [@sutskever2014sequence]. The most widely adopted training
method for sequence-to-sequence models is maximum likelihood estimation
\[MLE\], where decoding is done by predicting the output to which the
model assigns maximum likelihood. However, as the models predict tokens
one by one, exact search is not feasible in the general case, and the
community has resorted to using heuristics instead. The most popular of
these heuristics is beam search which has been shown to have severe
flaws over the years. [@stahlberg-byrne-2019-nmt] showed that the model
assigns the highest score to the empty sentence in greater than 50% of
the cases and that search errors are more frequent than model errors, in
addition to being more difficult to diagnose and fix.
[@welleck-etal-2020-consistency] found that a sequence which receives
zero probability under a recurrent language model's distribution can
receive non-zero probability under the distribution induced by the
decoding algorithm. [@stahlberg-byrne-2019-nmt] provide a possible
explanation for the MT community's continuing use of beam search despite
its flaws: search errors in beam search decoding, paradoxically, prevent
the decoder from choosing the empty hypothesis, which often gets the
global best model score as a side-effect of using maximum likelihood
estimation.

### Robustness and adversarial attacks 

Like most other deep learning models, NMT models have been found to be
sensitive to synthetic and natural noise [@DBLP:conf/iclr/BelinkovB18],
distributional shift and adversarial examples
[@heigold-etal-2018-robust]. Real-world MT systems need to deal with
increasingly non-standard and noisy text found on the internet but
absent from many standard benchmark datasets. Machine translation
robustness featured as a shared task in the WMT 2020 challenge
[@specia2020findings] where MT systems were evaluated in zero-shot and
few-shot scenarios to test for robustness. All accepted submissions
trained their systems using big-transformer models, boosted performance
with tagged back-translation, continued training with filtered and
in-domain data, and assembled ensembles of different models to improve
performance.

The increasing body of work on adversarial examples has shown the
potential hazards of employing brittle machine learning systems so
widely in practical applications
[@DBLP:journals/corr/GoodfellowSS14; @narodytska2017simple; @sakaguchi2017robsut].
[@anastasopoulos-etal-2019-neural] focus on the grammatical errors made
by non-native speakers and show that augmenting training data with
sentences containing artificially introduced grammatical errors can make
the system more robust to such errors. [@DBLP:conf/iclr/BelinkovB18]
show that character-based NMT models break down when presented with both
natural and synthetic noise. They also demonstrate that synthetic noise
does not capture a lot of the variation present in natural noise
resulting in models that perform poorly while translating natural noise.
[@heigold-etal-2018-robust] evaluate the robustness of NMT systems
against perturbed word forms that do not pose a challenge to humans and
corroborate the finding that training on noisy data can help models
achieve improved performance on noisy data.

### Bias 

Natural language training data inevitably reflects the biases and
stereotypes present in our society. Systems trained on this biased data
often reflect or even amplify these biases and their harmful
stereotypes. [@prates2020assessing] showed that translating sentences
from gender-neutral languages to English using Google Translate
exhibited gender biases and a strong tendency toward male defaults.
Google Translate now adds feminine and masculine forms for translated
sentences, partially addressing some of the shortcomings mentioned in
the paper. [@saunders-byrne-2020-reducing] proposed treating gender
debiasing as a [domain adaptation](#heading:domadapt) problem making use
of the extensive literature in domain adaptation for NMT systems. They
demonstrate improved debiasing without degradation in overall
translation quality by transfer learning on a small set of trusted,
gender-balanced examples.

## Possible Areas of Future Work

### Large Language Models 

Transformers have changed the zeitgeist of MT research from
fully-supervised learning to *pre-train and fine-tune* and now to
*pre-train and prompt*. Large language models (LLMs) can now be prompted
to perform very high-quality machine translation (MT), even though they
were not explicitly trained for this task.
[@ghazvininejad2023dictionary] propose using a dictionary to identify
rare words or phrases in the source language and then generating prompts
that provide additional context for these words or phrases, which are
then used to guide the LLM to generate more accurate translations. The
authors demonstrate the effectiveness of this approach by evaluating it
on several language pairs and showing significant improvements in
machine translation performance.

Despite its great potential, prompt-based learning faces several
challenges. [@zhang2023prompting] demonstrate that sometimes prompting
results in the *rejection* of the input where the LLM responds in the
wrong target language, under-translates the input, mistranslates
entities like dates, or even just copies source phrases. In addition to
the general limitations of LLMs, such as hallucination, the authors also
observed a phenomenon specific to prompting, which they call the 'prompt
trap'. This occurs when translations are heavily influenced by the
prompt or the prefix of the source template leading to suboptimal or
incorrect translations. Empirical evidence suggests that the performance
of an LLM depends on both the templates being used and the answers being
considered. However, finding the best combination of template and answer
simultaneously through search or learning remains a challenging research
question [@liu2023pre].

### Multilingual 

Achieving human-level universal translation between all possible natural
language pairs is the holy grail of machine translation research.
Multilingual NMT \[MNMT\] systems are highly desirable as they can be
trained with data from various language pairs, which can aid
resource-poor languages in acquiring extra knowledge from other
languages [@shatz2017native]. Furthermore, MNMT systems tend to exhibit
better generalisation capabilities due to their exposure to diverse
languages resulting in improved translation quality compared with
bilingual NMT systems in a phenomenon referred to as 'translation
knowledge transfer' [@5288526]. [@fan2021beyond] proposed M2M-100, a
Many-to-Many multilingual translation model capable of translating
between the 9,900 directions of 100 languages. The authors employed both
dense and sparse scaling techniques by introducing language-specific
parameters trained with a novel random re-routing scheme. Their model
outperforms an English-centric baseline by more than 10 BLEU points on
average when translating directly between non-English directions.

Current MNMT approaches experience difficulties incorporating over 100
language pairs without sacrificing translation quality---incremental
learning and knowledge distillation show promise in addressing this
issue. Translating multilingualism *within* a sentence, such as
code-mixed input and output, creoles, and pidgins, is an exciting
research direction as compact MNMT models can handle code-mixed input,
but code-mixed output is still an open problem
[@johnson-etal-2017-googles].

### Document-level 

Despite its success, machine translation has been based mainly on strong
independence and locality assumptions. This means that sentences are
translated in isolation, independent of their document-level
inter-dependencies. However, text is made up of collocated and
structured groups of sentences bound together by complex linguistic
elements, referred to as 'discourse' [@slp2e]. Moreover, ambiguous words
in a sentence can only be disambiguated by their surrounding context. A
recent paper by [@liu-etal-2020-multilingual-denoising] illustrates this
research direction. The authors corrupt input documents by masking
phrases and permuting sentences, resulting in input sequences up to 512
tokens and then train a single Transformer model to recover the original
monolingual document segments. By using document fragments, the model is
able to learn long-range dependencies between sentences and outperform
sentence-level NMTs. However, it was also observed that without
pre-training, document-level NMT models perform much worse than their
sentence-level counterparts, suggesting that pre-training is a crucial
step and a promising strategy for improving document-level NMT
performance.

Despite promising results, document-level NMTs face multiple challenges
[@maruf2021survey]. Existing metrics like BLEU and METEOR do not account
for specific discourse phenomena in the translation, which can lead to
failures in evaluating the quality of longer pieces of generated text.
Most methods only use a small context beyond a single sentence, which
consists of neighbouring sentences and do not incorporate context from
the whole document. Additionally, more research is required to determine
whether the global context is truly beneficial to improve translation
performance.

## Conclusion

The field of machine translation is rapidly evolving, with many exciting
developments in areas such as large language models, multilingual
translation, and document-level translation. While many challenges
remain to be addressed, including robustness, bias and lack of data for
under-represented languages, the potential for machine translation to
bridge language barriers and facilitate communication between people
worldwide is immense. Continued research and innovation in the field
will be crucial to unlocking this potential and creating more effective
and accurate machine translation systems.

*To have another language is to possess a second soul.*\
-- Charlemagne

[^1]: Here, *domain* is defined by a corpus from a specific source and
    may differ from other *domains* in topic, genre, or style

## References
