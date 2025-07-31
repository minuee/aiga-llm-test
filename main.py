import os
import pymysql
import re
import json

from dotenv import load_dotenv, dotenv_values
from runModel import evaluateDoctor

from logger import log

from database import cur, conn

# # .env 파일을 로드하고 변수 가져오기
# env_vars = dotenv_values(".env")

# # .env 파일에 정의된 모든 환경 변수를 삭제
# for var in env_vars:
#     if var in os.environ:
#         del os.environ[var]

# # print("환경 변수 삭제 후:", os.environ)

if 'RANGE_FROM_REVIEW_ID' in os.environ:
    del os.environ['RANGE_FROM_REVIEW_ID']
if 'RANGE_SNS_URL' in os.environ:
    del os.environ['RANGE_SNS_URL']
if 'RANGE_LIMIT' in os.environ:
    del os.environ['RANGE_LIMIT']

load_dotenv(override=True)

DATA_VERSION_ID = os.getenv('DATA_VERSION_ID')

def contains_any_substring(text, substrings):
    return any(substring in text for substring in substrings)

def main():
    range_base_date = os.getenv('RANGE_BASE_DATE')

    range_from_review_id = os.getenv('RANGE_FROM_REVIEW_ID')
    range_sns_url  = os.getenv('RANGE_SNS_URL')
    range_limit    = os.getenv('RANGE_LIMIT')

    SOURCE_ID = os.getenv('SOURCE_ID')


    if SOURCE_ID == 'naver_cafe':
        base_str = """WITH distinct_sns AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY snsUrl, articleNo ORDER BY review_id) AS rn
            FROM patient_review where data_version_id = '%s' and source_id = '%s' and articleDate > '%s' 
        )
        SELECT  review_id, title, contents, comments, articleLink, articleNo, articleDate, source_id, snsName, snsUrl 
        FROM distinct_sns
        WHERE rn = 1 and contents like '{%%' 
        """ % (DATA_VERSION_ID, SOURCE_ID, range_base_date)
    elif SOURCE_ID == 'naver_kin':
        base_str = """WITH distinct_sns AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY title, articleDate ORDER BY review_id) AS rn
            FROM patient_review where data_version_id = '%s' and source_id = '%s' and articleDate > '%s' 
        )
        SELECT  review_id, title, contents, comments, articleLink, articleNo, articleDate, source_id, snsName, snsUrl 
        FROM distinct_sns
        WHERE rn = 1 and contents like '{%%' 
        """ % (DATA_VERSION_ID, SOURCE_ID, range_base_date)
    elif SOURCE_ID == 'youtube':            # exactly: 1:의료전문채널, 0: 병원
        base_str = """WITH distinct_sns AS (
            SELECT *,
                ROW_NUMBER() OVER (PARTITION BY snsName, title, articleDate ORDER BY review_id) AS rn
            FROM patient_review where data_version_id = '%s' and source_id = '%s' and articleDate > '%s' and comments is not null
        )
        SELECT  review_id, title, contents, comments, articleLink, articleNo, articleDate, source_id, snsName, snsUrl 
        FROM distinct_sns
        WHERE rn = 1 and contents like '{%%' 
        """ % (DATA_VERSION_ID, SOURCE_ID, range_base_date)

    sid_str     = f"and review_id > {range_from_review_id} " if range_from_review_id else ""
    sns_url_str = f"and snsUrl = '{range_sns_url}' " if range_sns_url else ""
    limit_str   = f"limit {range_limit} " if range_limit else ""
    query = base_str
    query += sid_str
    query += sns_url_str
    query += "order by review_id "
    query += limit_str


    # query = """select review_id, title, contents, comments,
    # articleLink, articleNo, articleDate, source_id, snsName, snsUrl 
    # from sns_recommand 
    # where 
    # contents like '{%%' and articleDate > '%s' and review_id > %s 
    # and (snsUrl = '%s') >
    # limit %s
    # """ % (range_base_date, range_from_review_id, range_sns_url, range_limit)

    # query = """
    # select review_id, title, contents, comments, 
    # articleLink, articleNo, articleDate, source_id, snsName, snsUrl from sns_recommand 
    # where review_id in (182549)
    # """  

    # ,30292,41262,47067)
    # (20176,133288,136557,140733,140807,144199,144292,144632,162137,162206,169194,169306)
    # (29802,30292,41262,47067,62775,62777,62851,100471,100477,112010,116018,119196,120483,127837,138332,138416,142398,147121)

    # print('query:', query)
    log.LogTextOut(f"query: {query}")

    cur.execute(query)

    patient_reviews = []

    while (True):
        row = cur.fetchone()
        if row==None:
            break
        
        review_id   = row[0]
        title       = row[1]
        contents    = row[2]
        comments    = row[3]
        articleLink = row[4]
        articleNo   = row[5]
        articleDate = row[6]
        source_id   = row[7] 
        snsName     = row[8]
        snsUrl      = row[9]

        try:
            list_comments = json.loads(comments)
        except json.JSONDecodeError as e:
            print(f"Error parsing comments: {e}")
            list_comments = []
        
        if SOURCE_ID == 'youtube': 
            filtered_dict = []
            for comment in list_comments:
                # 채널에서 작성한 댓글은 제외
                if comment['writer'] in snsUrl:
                    continue
                substrings = ['교수','의사','수술','진료','추천']
                result = contains_any_substring(comment['comment'], substrings)
                # 키워드가 없는 댓글도 제외
                if result == False:
                    continue
                filtered_dict.append(comment)

            if len(filtered_dict) < 1:
                continue
            
            list_comments = str(filtered_dict)

        row = {
            "review_id":review_id, 
            "title":title, 
            "contents":contents, 
            "comments":list_comments, 
            "articleLink":articleLink, 
            "articleNo":articleNo, 
            "articleDate":articleDate,
            "source_id":source_id,
            "snsName":snsName, 
            "snsUrl":snsUrl
        }
        # print('row:', row)
        patient_reviews.append(row)
        
    # print("patient_reviews:", patient_reviews)
    if len(patient_reviews) == 0:
        return

    # doctor_evals = evaluateDoctor(patient_reviews)
    evaluateDoctor(patient_reviews)

    # print("doctor_evals:", doctor_evals)   

    # for doctor_eval in doctor_evals:
    #     for sns in patient_reviews:
    #         if sns["review_id"] == doctor_eval["review_id"]:
    #             print('저장', sns["review_id"])
    #             saveEvaluation(sns, doctor_eval)

    msg = f"처리된 범위 from: {patient_reviews[0]['review_id']}, end: {patient_reviews[-1]['review_id']}"
    print(msg)
    conn.close()

if __name__ == '__main__':
    log.LogTextOut(f"main() 실행")
    main()
    print("done!")