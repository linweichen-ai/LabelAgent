import io,json,unittest,urllib.error
from tools.sam_busy_retry import post_when_available,BUSY_MESSAGE

class Opener:
    def __init__(self,results):self.results=iter(results);self.bodies=[]
    def open(self,request,timeout):
        self.bodies.append(request.data);value=next(self.results)
        if isinstance(value,Exception):raise value
        return io.BytesIO(json.dumps(value).encode())

def error(message,status=503):
    return urllib.error.HTTPError('http://127.0.0.1:8766/segment_text',status,'error',{},io.BytesIO(json.dumps({'error':{'message':message}}).encode()))

class BusyRetryTests(unittest.TestCase):
    def test_only_busy_rejection_is_retried_and_body_is_identical(self):
        opener=Opener([error(BUSY_MESSAGE),{'results':[]}]);delays=[];busy=[]
        value=post_when_available(opener,'http://127.0.0.1:8766/segment_text',b'original',clock=lambda:0,sleep=delays.append,on_busy=lambda:busy.append(1))
        self.assertEqual(value,{'results':[]});self.assertEqual(opener.bodies,[b'original',b'original'])
        self.assertEqual(busy,[1]);self.assertEqual(delays,[1])

    def test_other_503_is_not_retried(self):
        for payload in ({'error':{'message':'worker failed'}},{'error':'worker failed'},[]):
            failure=urllib.error.HTTPError('http://127.0.0.1:8766/segment_text',503,'error',{},io.BytesIO(json.dumps(payload).encode()))
            opener=Opener([failure])
            with self.assertRaises(urllib.error.HTTPError):post_when_available(opener,'http://127.0.0.1:8766/segment_text',b'original')
            self.assertEqual(len(opener.bodies),1)

    def test_unknown_timeout_is_not_retried(self):
        opener=Opener([TimeoutError('unknown acceptance')])
        with self.assertRaises(TimeoutError):post_when_available(opener,'http://127.0.0.1:8766/segment_text',b'original')
        self.assertEqual(len(opener.bodies),1)

    def test_busy_deadline_stops_without_an_accepted_repeat(self):
        times=iter((0,2));opener=Opener([error(BUSY_MESSAGE)])
        with self.assertRaises(RuntimeError):post_when_available(opener,'http://127.0.0.1:8766/segment_text',b'x',busy_wait=1,clock=lambda:next(times))
        self.assertEqual(len(opener.bodies),1)

if __name__=='__main__':unittest.main()
