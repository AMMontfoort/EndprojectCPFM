# Predicting Bitcoin, Ethereum and XRP prices

# Introduction
With Bitcoin reaching a price higher than 100.000 dollars in 2024, its all-time peak, it seems lucrative to invest in cryptocurrencies (Yahoo Finance, n.d.). Rather than buying low and selling high, a more strategic approach using mathematical models or so-called algorithmic trading is used in the report. This report discusses a trading strategy based on the prediction of cryptocurrencies prices using a neural network (NN). The advantage of NNs is their ability to analyze complex relationships without needing detailed knowledge of the underlying structure (Penm, Chaar \& Moles, 2013). However, this ability also presents a disadvantage: deep learning models are less interpretable than classical mathematical models and are often considered "black-box" models, making it challenging to understand the specific relationships they learn between inputs and outputs.

The development of NNs began in 1943, when neurophysiologist McCulloch and mathematician Pitts created an early NN model using electrical circuits (Foote, 2021). This research was further developed by psychologist Hebb, who discovered that repeated stimulation of a neuron could strengthen its response (Foote, 2021). NNs are designed to mimic the human brain, which is a network consisting of interconnected neurons. The human brain learns by repetitively being exposed to the same stimulus and therefore altering the strength of synaptic connections between neurons. Similarly, NNs learn by adjusting the weights between neurons until the network's output closely matches the desired target value. 

The overall goal of this research is to get a clear methodology to predict the price of cryptocurrencies based on a NN, so that insights into buying and selling opportunities are given. In this report, the cryptocurrency prices of Bitcoin, Ethereum and XRP are predicted while minimizing the error or loss of the prediction. The trading strategy focuses solely on if there is a trading opportunity (if it is attractive to buy or sell) and not how much should be traded.

The remainder of this report is structured as follows: Section Literature provides a detailed theoretical background on the methods and algorithms, Section Algorithm & Python code describes the design, implementation and tutorial of the algorithm, Section Data summarizes the used data, Section Results describes the main findings and Section Conclusion & discussion summarizes insights and conclusions drawn from the results.

# Literature
# Structure of the NN
This Section discusses the Neural Network (NN) in general form based on Hastie, Tibshirani and Friedman (2009), Nielsen (2015) and Taddy (2019). A neural network consists, similar to a human brain,
of several neurons. In this report a multilayer NN consists of an input layer of d normalized inputs, L hidden layers and an output layer of 1 output.

\subsection{Activation function}
\label{subsection: Activation function}
As described in Section \ref{subsection: forward pass}, this report investigates activation functions hyperbolic tangent (\ref{equation:hyperbolic tangent}), sigmoid (\ref{equation:sigmoid}) and rectified linear unit (\ref{equation:ReLu}):
\begin{flalign}
\label{equation:hyperbolic tangent}
\delta(x) = \frac{\exp{(x)}-\exp{(-x)}}{\exp{(x)}+\exp{(-x)}}  &&
\end{flalign}
\begin{flalign}
\label{equation:sigmoid}
\delta(x) = \frac{1}{1+\exp{(-x)}}  &&
\end{flalign}
\begin{flalign}
\label{equation:ReLu}
\delta(x) = \max(0,x)  &&
\end{flalign}
