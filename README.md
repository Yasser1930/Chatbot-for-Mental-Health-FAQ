# Chatbot-for-Mental-Health-FAQ

This repository contains a dataset and a complete workflow for analyzing frequently asked questions (FAQs) about mental health. This project includes data preprocessing, visualization, and a machine-learning model that can provide answers to mental health questions by classifying and predicting responses.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [TF-IDF Vectorization](#tf-idf-vectorization)
- [Usage](#usage)
- [Features](#features)

## Project Overview

This project leverages natural language processing (NLP) techniques to analyze a mental health FAQ dataset and provides an end-to-end pipeline from data cleaning to model deployment. The primary goal is to create a model that can answer frequently asked mental health questions by classifying and matching relevant responses.

## Dataset

The `Mental_Health_FAQ.csv` dataset includes:
- **Question_ID**: A unique identifier for each question-answer pair.
- **Questions**: Text of the mental health-related questions.
- **Answers**: Detailed responses to each question.

## Approach

The following approach was taken to process, analyze, and model the FAQ data:

1. **Data Loading and Initial Exploration**: Load and inspect the dataset for initial understanding.
2. **Text Preprocessing**: Apply various cleaning functions, including lowercasing, contraction expansion, removing stopwords, and lemmatization, to prepare text for analysis.
3. **Feature Extraction with TF-IDF**: Transform the text data using TF-IDF vectorization.
4. **Visualization**: Generate bar plots and word clouds to highlight common words in questions and answers.
5. **Machine Learning Model**: Train a Linear Support Vector Classifier (SVC) model to predict answers based on questions.

## TF-IDF Vectorization

Term Frequency-Inverse Document Frequency (TF-IDF) is used in this project to transform textual data into numerical form for machine learning. It is a popular technique for text representation, particularly in NLP, and works by calculating two main metrics for each term in a document:

- **Term Frequency (TF)**: Measures how frequently a word appears in a document.
- **Inverse Document Frequency (IDF)**: Measures the importance of a word by considering how commonly it appears across all documents in the dataset.

Together, TF-IDF highlights terms that are more informative and unique to specific questions rather than common words that appear frequently across all questions. This improves the relevance of terms by down-weighting commonly used words (e.g., "mental", "health") and up-weighting unique terms, thus providing more context for machine learning models to understand each question better.

In this project:
- **TfidfVectorizer** from `scikit-learn` is used to transform the questions into TF-IDF vectors.
- These vectors serve as features for training the Linear Support Vector Classifier (SVC) model, enabling the model to predict answers based on the unique language patterns in questions.

## Usage:
1- **Run the Preprocessing Pipeline**:
  - Load and preprocess the dataset by applying the full_preprocess() function to clean the text data.
2- **Explore the Dataset**:
  - Use the data exploration section to visualize top words in questions and answers through bar plots and word clouds.
3- **Train and Test the Model**:
  - Train the LinearSVC model on the processed data and evaluate it using example questions. 

## Features:
- Text Preprocessing: Clean and preprocess questions and answers using advanced NLP techniques.
- Data Visualization: Visualize common themes in questions and answers with frequency plots and word clouds.
- Question-Answer Prediction: Use a trained LinearSVC model to predict answers based on new questions.
