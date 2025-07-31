import os
import pymysql

from dotenv import load_dotenv, dotenv_values

from logger import log

load_dotenv(override=True)

DATA_VERSION_ID = os.getenv('DATA_VERSION_ID')

db_host = os.environ['MYSQL_HOST']
db_user = os.environ['MYSQL_USER']
db_pwd = os.environ['MYSQL_PASSWORD']
db_database = os.environ['MYSQL_DATABASE']
db_post = os.environ['MYSQL_PORT']


print('\n db_host:', db_host)
print('\n db_hodb_userst:', db_user)
print('\n db_pwd:', db_pwd)
print('\n db_database:', db_database)

conn = pymysql.connect(host=db_host, user=db_user, password=db_pwd, db=db_database, charset='utf8')
cur = conn.cursor()

SOURCE_ID = os.getenv('SOURCE_ID')

def saveEvaluation(sns, doctor_eval):
    try: 
        review_id         = int(sns["review_id"])
        # contents    = sns["contents"]
        # comments    = sns["comments"]
        # snsName     = sns["snsName"]
        # snsUrl      = sns["snsUrl"]
        # articleLink = sns["articleLink"]
        # articleNo   = int(sns["articleNo"])
        # articleDate = sns["articleDate"]
        # source_id     = sns["source_id"]
        # title       = sns["title"]

        if "eval" in doctor_eval:
            d_eval = doctor_eval["eval"]
            article_type                    = d_eval["type"]
            writer                          = ','.join(d_eval["writer"]) if isinstance(d_eval["writer"], list) else d_eval["writer"]
            doctor                          = d_eval["doctor"]
            hospital                        = d_eval["hospital"]
            department                      = d_eval["Department"]
            desease                         = d_eval["disease"] if "disease" in d_eval else None
            evaluation = d_eval["evaluation"]
            if SOURCE_ID == 'naver_cafe' or SOURCE_ID == 'youtube':
                kindness_score                  = int(evaluation["Kindness and consideration"]["score"])
                kindness_confidence             = evaluation["Kindness and consideration"]["confidence"]
                kindness_evidence               = evaluation["Kindness and consideration"]["evidence"] if "evidence" in evaluation["Kindness and consideration"] else None
                satisfaction_score              = int(evaluation["Treatment satisfaction"]["score"])
                satisfaction_confidence         = evaluation["Treatment satisfaction"]["confidence"]
                satisfaction_evidence           = evaluation["Treatment satisfaction"]["evidence"] if "evidence" in evaluation["Treatment satisfaction"] else None
                detailedExplanation_score       = int(evaluation["Clear concise explanation"]["score"])
                detailedExplanation_confidence  = evaluation["Clear concise explanation"]["confidence"]
                detailedExplanation_evidence    = evaluation["Clear concise explanation"]["evidence"] if "evidence" in evaluation["Clear concise explanation"] else None
                recommand_score                 = int(evaluation["Doctor recommendation"]["score"])
                recommand_confidence            = evaluation["Doctor recommendation"]["confidence"]
                recommand_evidence              = evaluation["Doctor recommendation"]["evidence"] if "evidence" in evaluation["Doctor recommendation"] else None
            elif SOURCE_ID == 'naver_kin':
                kindness_score                  = None
                kindness_confidence             = None
                kindness_evidence               = None
                satisfaction_score              = None
                satisfaction_confidence         = None
                satisfaction_evidence           = None
                detailedExplanation_score       = None
                detailedExplanation_confidence  = None
                detailedExplanation_evidence    = None
                recommand_score                 = int(evaluation["Doctor recommendation"]["score"])
                recommand_confidence            = evaluation["Doctor recommendation"]["confidence"]
                recommand_evidence              = evaluation["Doctor recommendation"]["evidence"] if "evidence" in evaluation["Doctor recommendation"] else None                 

                # 작성자 본인을 추천하는 경우는 점수 0으로 고정
                if writer == doctor: 
                    recommand_score = 0

        else:
            # 중복 처리 방지를 위해 UNIQUE KEY('NULL')로 설정
            article_type                    = 'NULL'
            writer                          = 'NULL'
            doctor                          = 'NULL'
            hospital                        = 'NULL'
            department                      = None
            desease                         = None
            kindness_score                  = None
            kindness_confidence             = None
            kindness_evidence               = None
            satisfaction_score              = None
            satisfaction_confidence         = None
            satisfaction_evidence           = None
            detailedExplanation_score       = None
            detailedExplanation_confidence  = None
            detailedExplanation_evidence    = None
            recommand_score                 = None
            recommand_confidence            = None
            recommand_evidence              = None

        # query = """INSERT INTO sns_evaluation (sid, snsSite, articleDate, 
        #     snsName, snsUrl, articleLink, articleNo, title,
        #     type, writer, doctor, hospital, department, 
        #     desease, kindness_score, kindness_evidence, satisfaction_score, satisfaction_evidence,
        #     detailedExplanation_score, detailedExplanation_evidence, recommand_score, recommand_evidence, 
        #     createdate, updatedate
        #     )
        #     VALUES ( %s, %s, %s, 
        #     %s, %s, %s, %s, %s, 
        #     %s, %s, %s, %s, %s, 
        #     %s, %s, %s, %s, %s, 
        #     %s, %s, %s, %s, 
        #     now(), NULL)
        #     """
        # cur.execute(query, ( sid, snsSite, articleDate, 
        #             snsName, snsUrl, articleLink, articleNo, title,
        #             article_type, writer, doctor, hospital, department,
        #             desease, kindness_score, kindness_evidence, satisfaction_score, satisfaction_evidence,
        #             detailedExplanation_score, detailedExplanation_evidence, recommand_score, recommand_evidence
        # ))
        cur.callproc('set_review_evaluation', [
                        review_id, 
                        DATA_VERSION_ID, 
                        article_type, writer, doctor, hospital, department, desease, 
                        kindness_score, 
                        kindness_confidence,
                        kindness_evidence,
                        satisfaction_score, 
                        satisfaction_confidence, 
                        satisfaction_evidence,
                        detailedExplanation_score, 
                        detailedExplanation_confidence, 
                        detailedExplanation_evidence,
                        recommand_score, 
                        recommand_confidence, 
                        recommand_evidence
                    ]
                    )
        for result in cur.fetchall():
            print(result)

        print('저장 review_id:', sns["review_id"])
        
        conn.commit()
    except Exception as e:
            msg = f"Error, review_id({sns['review_id']}), in saveEvaluation: {e}"
            log.LogTextOut(msg)