"""Retry only a manager rejection that occurred before model activation."""
import json,time,urllib.error,urllib.request
BUSY_MESSAGE='Another model request is active; retry after it finishes'

def post_when_available(opener,url,body,*,timeout=600,busy_wait=300,pause=1,on_busy=None,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+busy_wait
    while True:
        request=urllib.request.Request(url,body,headers={'Content-Type':'application/json'})
        try:
            with opener.open(request,timeout=timeout) as response:return json.load(response)
        except urllib.error.HTTPError as error:
            raw=error.read()
            try:value=json.loads(raw)
            except (ValueError,TypeError):value={}
            details=value.get('error') if isinstance(value,dict) else None
            message=details.get('message') if isinstance(details,dict) else None
            if error.code!=503 or message!=BUSY_MESSAGE:raise
            now=clock()
            if now>=deadline:raise RuntimeError('Model manager remained busy; no accepted request was retried') from error
            if on_busy:on_busy()
            sleep(min(pause,deadline-now))
