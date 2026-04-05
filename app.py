import nltk
nltk.download('vader_lexicon')

from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)
sid = SentimentIntensityAnalyzer()


def review_rating(string):
    scores = sid.polarity_scores(string)

    original_pos = scores['pos']
    original_neg = scores['neg']
    original_neu = scores['neu']
    compound = scores['compound']

    # Emotional intensity
    emotional_intensity = abs(compound)

    # Improved neutral reduction logic
    if emotional_intensity > 0.7:
        neutral_reduction = 0.6
    elif emotional_intensity > 0.5:
        neutral_reduction = 0.45
    elif emotional_intensity > 0.3:
        neutral_reduction = 0.3
    elif emotional_intensity > 0.1:
        neutral_reduction = 0.15
    else:
        neutral_reduction = 0

    # Reduce neutral
    reduced_neutral = original_neu * neutral_reduction
    adjusted_neu = original_neu - reduced_neutral

    # ✅ Distribute neutral proportionally (FIXED LOGIC)
    total_sentiment = original_pos + original_neg

    if total_sentiment > 0:
        pos_share = original_pos / total_sentiment
        neg_share = original_neg / total_sentiment
    else:
        # Edge case: fully neutral text
        pos_share = 0.5
        neg_share = 0.5

    adjusted_pos = original_pos + (reduced_neutral * pos_share)
    adjusted_neg = original_neg + (reduced_neutral * neg_share)

    # Normalize to 100%
    total = adjusted_pos + adjusted_neg + adjusted_neu

    if total > 0:
        positive_percentage = (adjusted_pos / total) * 100
        negative_percentage = (adjusted_neg / total) * 100
        neutral_percentage = (adjusted_neu / total) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0
        neutral_percentage = 100

    return {
        "Positive": f"{positive_percentage:.1f}%",
        "Negative": f"{negative_percentage:.1f}%",
        "Neutral": f"{neutral_percentage:.1f}%",
        "positive_raw": positive_percentage,
        "negative_raw": negative_percentage,
        "neutral_raw": neutral_percentage,
        "compound": compound,
        "emotional_intensity": emotional_intensity
    }


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form['review']
        result = review_rating(review)
        return render_template('index.html', review=review, result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
