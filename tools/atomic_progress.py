"""Bounded atomic JSON publication through transient Windows read locks."""
import json,os,threading,time
from pathlib import Path

def save_json(path,value,attempts=20,delay=.025):
    if type(attempts) is not int or attempts<1 or delay<0:raise ValueError('Positive attempts and nonnegative delay required')
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
    for index in range(attempts):
        try:
            os.replace(temporary,path)
            return
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32) and error.errno!=13:raise
            if index+1==attempts:raise
            time.sleep(delay)
