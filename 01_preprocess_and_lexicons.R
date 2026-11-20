# =====================================================================
# 01_preprocess_and_lexicons.R
# Sentiment analysis assignment - Part 1 (R)
# Steps: load data -> preprocess -> apply AFINN, Bing, NRC -> save CSV
# Dataset: IMDB Dataset of 50K Movie Reviews (Kaggle, Lakshmipathi N)
#   columns: review, sentiment   -> save the file as "IMDB Dataset.csv"
#   in your working directory.
# =====================================================================

# ---- 0. Packages (run install.packages once if missing) --------------
# install.packages(c("tidyverse", "tidytext", "textdata"))
library(tidyverse)   # dplyr, ggplot2, readr, stringr
library(tidytext)    # unnest_tokens, get_sentiments, stop_words
library(textdata)    # needed by tidytext for AFINN and NRC

# NOTE: the FIRST time you call get_sentiments("afinn") / ("nrc"),
# R asks "Is it OK to download? 1: Yes". Type 1 and press Enter.
# Run these two lines once in the console before running the script:
#   get_sentiments("afinn"); get_sentiments("nrc")

set.seed(42)

# ---- 1. Data collection ---------------------------------------------
raw <- read_csv("IMDB Dataset.csv", show_col_types = FALSE)

# Take a balanced random sample of 2000 reviews (1000 pos / 1000 neg)
# to keep the analysis fast
reviews <- raw %>%
  group_by(sentiment) %>%
  slice_sample(n = 1000) %>%
  ungroup() %>%
  slice_sample(prop = 1) %>%                 # shuffle
  mutate(id = row_number()) %>%
  rename(original_label = sentiment)

cat("Dataset size:", nrow(reviews), "reviews\n")
print(table(reviews$original_label))

# ---- 2. Preprocessing -----------------------------------------------
reviews <- reviews %>%
  mutate(
    text_clean = review %>%
      str_remove_all("<[^>]+>") %>%          # remove HTML tags such as <br />
      str_to_lower() %>%                     # lowercase
      str_replace_all("[^a-z\\s]", " ") %>%  # remove punctuation, digits, symbols
      str_squish()                           # collapse extra whitespace
  )

# Tokenisation: one row per word
tokens <- reviews %>%
  select(id, text_clean) %>%
  unnest_tokens(word, text_clean)            # tokenisation

cat("Tokens before stop-word removal:", nrow(tokens), "\n")

# Stop-word removal using tidytext's built-in stop_words list
tokens <- tokens %>% anti_join(stop_words, by = "word")

cat("Tokens after stop-word removal:", nrow(tokens), "\n")

# Rebuild a cleaned text per review (used later by Python)
clean_text <- tokens %>%
  group_by(id) %>%
  summarise(clean_text = paste(word, collapse = " "), .groups = "drop")

# ---- 3a. AFINN lexicon ----------------------------------------------
# Each word has an integer score from -5 (very negative) to +5 (very positive).
# Document score = sum of word scores.
afinn_scores <- tokens %>%
  inner_join(get_sentiments("afinn"), by = "word") %>%
  group_by(id) %>%
  summarise(afinn_score = sum(value), .groups = "drop")

# ---- 3b. Bing lexicon -----------------------------------------------
# Each word is labelled positive or negative.
# Document score = (# positive words) - (# negative words).
bing_scores <- tokens %>%
  inner_join(get_sentiments("bing"), by = "word", relationship = "many-to-many") %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(bing_score = positive - negative) %>%
  select(id, bing_pos = positive, bing_neg = negative, bing_score)

# ---- 3c. NRC lexicon ------------------------------------------------
# Each word can carry several categories: 2 polarities (positive, negative)
# and 8 emotions (anger, anticipation, disgust, fear, joy, sadness,
# surprise, trust).
nrc <- get_sentiments("nrc")

# (i) polarity score: positive - negative
nrc_polarity <- tokens %>%
  inner_join(nrc %>% filter(sentiment %in% c("positive", "negative")),
             by = "word", relationship = "many-to-many") %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
  mutate(nrc_score = positive - negative) %>%
  select(id, nrc_score)

# (ii) emotion counts per review (for the sample table in the report)
nrc_emotions <- tokens %>%
  inner_join(nrc %>% filter(!sentiment %in% c("positive", "negative")),
             by = "word", relationship = "many-to-many") %>%
  count(id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

# ---- 4. Combine and save --------------------------------------------
results <- reviews %>%
  select(id, original_label) %>%
  left_join(clean_text,   by = "id") %>%
  left_join(afinn_scores, by = "id") %>%
  left_join(bing_scores,  by = "id") %>%
  left_join(nrc_polarity, by = "id") %>%
  mutate(across(c(afinn_score, bing_score, nrc_score), ~ replace_na(.x, 0)),
         clean_text = replace_na(clean_text, ""))

# Convert scores to labels (>0 positive, <0 negative, 0 neutral)
to_label <- function(x) case_when(x > 0 ~ "positive", x < 0 ~ "negative", TRUE ~ "neutral")
results <- results %>%
  mutate(afinn_label = to_label(afinn_score),
         bing_label  = to_label(bing_score),
         nrc_label   = to_label(nrc_score))

write_csv(results, "reviews_clean.csv")   # <- Python script reads this file

# ---- 5. Outputs to paste into the report ------------------------------
cat("\n--- Sample of lexicon outputs (first 10 reviews) ---\n")
print(results %>% select(id, original_label, afinn_score, afinn_label,
                         bing_score, bing_label, nrc_score, nrc_label) %>% head(10))

cat("\n--- Sample NRC emotion counts ---\n")
print(head(nrc_emotions, 10))

cat("\n--- Label distribution per lexicon ---\n")
print(table(AFINN = results$afinn_label))
print(table(Bing  = results$bing_label))
print(table(NRC   = results$nrc_label))

# Comparison: how well does each lexicon agree with the dataset's own labels?
# (only on reviews the lexicon did not call neutral)
agree <- function(pred) {
  keep <- pred != "neutral"
  mean(pred[keep] == results$original_label[keep])
}
comparison <- tibble(
  lexicon  = c("AFINN", "Bing", "NRC"),
  accuracy = c(agree(results$afinn_label), agree(results$bing_label), agree(results$nrc_label))
)
cat("\n--- Agreement with original IMDB labels ---\n")
print(comparison)

# Most common positive/negative words (Bing) - useful figure
top_words <- tokens %>%
  inner_join(get_sentiments("bing"), by = "word", relationship = "many-to-many") %>%
  count(word, sentiment, sort = TRUE) %>%
  group_by(sentiment) %>% slice_max(n, n = 10) %>% ungroup()

p <- ggplot(top_words, aes(reorder_within(word, n, sentiment), n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  scale_x_reordered() + coord_flip() + facet_wrap(~sentiment, scales = "free_y") +
  labs(x = NULL, y = "Count", title = "Top 10 positive and negative words (Bing)")
ggsave("bing_top_words.png", p, width = 8, height = 4)
