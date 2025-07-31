import logging
import logging.handlers
import os
import datetime
import threading
import time

#메인 경로 (코드가 있는 파일의 경로)
# main_dir = os.path.split(os.path.abspath(__file__))[0]

class logSave():
    def __init__(self, dir, logname) -> None:
        self.logname = logname
        self.dir = os.path.join(dir,logname)
        self.InitLogger()

    def InitLogger(self):
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        if os.path.exists(self.dir) == False :
            os.makedirs(self.dir)
        
        log_File = os.path.join(self.dir,  self.logname + ".log")
        timedfilehandler = logging.handlers.TimedRotatingFileHandler(filename=log_File, when='midnight', interval=1, encoding='utf-8')
        timedfilehandler.setFormatter(formatter)
        timedfilehandler.suffix = "%Y%m%d.log"

        self.logger = logging.getLogger(self.logname)
        self.logger.addHandler(timedfilehandler)
        self.logger.setLevel(logging.INFO)

        #실행 될 때 체크해서 한번 지우고, 하루 단위로 오래된 파일 삭제
        self.delete_old_files(self.dir, 7)

        now = datetime.datetime.now()
        self.toDay = "%04d-%02d-%02d" % (now.year, now.month, now.day)
        self.th_auto_delete = threading.Thread(target=self.on_auto_delete, daemon=True)
        self.th_auto_delete.start()

    def LogTextOut(self,  msg):
        self.logger.info(str(msg))

    def delete_old_files(self, path_target, days_elapsed):
        """path_target:삭제할 파일이 있는 디렉토리, days_elapsed:경과일수"""
        for f in os.listdir(path_target): # 디렉토리를 조회한다
            f = os.path.join(path_target, f)
            if os.path.isfile(f): # 파일이면
                timestamp_now = datetime.datetime.now().timestamp() # 타임스탬프(단위:초)
                # st_mtime(마지막으로 수정된 시간)기준 X일 경과 여부
                is_old = os.stat(f).st_mtime < timestamp_now - (days_elapsed * 24 * 60 * 60)
                if is_old: # X일 경과했다면
                    try:
                        os.remove(f) # 파일을 지운다
                        print(f, 'is deleted') # 삭제완료 로깅
                    except OSError: # Device or resource busy (다른 프로세스가 사용 중)등의 이유
                        print(f, 'can not delete') # 삭제불가 로깅
    
    def on_auto_delete(self):
        while True:
            now = datetime.datetime.now()
            day = "%04d-%02d-%02d" % (now.year, now.month, now.day)
            if self.toDay != day:
                self.toDay = day
                self.delete_old_files(self.dir, 7)
            time.sleep(3600) #한시간마다 체크

log = logSave("logs", logname="LLMTest")