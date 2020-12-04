from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import flask
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')
get_df = pd.read_csv('data/final_df.csv')
get_sdf = get_df[get_df['description'].notna()]
get_tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = get_tf.fit_transform(get_sdf['description'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
sdf = get_sdf.reset_index()
titles = sdf['title']
indices = pd.Series(sdf.index, index=sdf['title'])
all_titles = [sdf['title'][i] for i in range(len(sdf['title']))]

def get_final_recommendations(title):
    get_idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[get_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = sdf['title'].iloc[movie_indices]
    dat = sdf['release_date'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title','Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
            
    if flask.request.method == 'POST':
        m_name = flask.request.form['movie_name']
        m_name = m_name.title()
#        check = difflib.get_close_matches(m_name,all_titles,cutout=0.50,n=1)
        if m_name not in all_titles:
            return(flask.render_template('negative.html',name=m_name))
        else:
            result_final = get_final_recommendations(m_name)
            names = []
            dates = []
            for i in range(len(result_final)):
                names.append(result_final.iloc[i][0])
                dates.append(result_final.iloc[i][1])

            return flask.render_template('positive.html',movie_names=names,movie_date=dates,search_name=m_name)

if __name__ == '__main__':
    app.run()