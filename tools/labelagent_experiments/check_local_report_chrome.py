"""Use a separate headless Chrome to check every image and real video playback.

Only this helper's own Chrome process is closed. No user browser is touched.
Uses standard-library CDP websocket framing; no dependency installation.
"""
import base64,hashlib,json,os,socket,struct,subprocess,sys,time,urllib.request,urllib.parse
from pathlib import Path
BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df')

class CDP:
    def __init__(self,url):
        u=urllib.parse.urlparse(url);self.s=socket.create_connection((u.hostname,u.port),timeout=30);self.s.settimeout(30);key=base64.b64encode(os.urandom(16)).decode();self.s.sendall(('GET '+u.path+' HTTP/1.1\r\nHost: '+u.netloc+'\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\n\r\n').encode());header=b''
        while not header.endswith(b'\r\n\r\n'):header+=self.exact(1)
        assert b' 101 ' in header and base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()) in header;self.n=0
    def exact(self,n):
        b=b''
        while len(b)<n:
            part=self.s.recv(n-len(b))
            if not part:raise EOFError('CDP closed')
            b+=part
        return b
    def send(self,payload):
        data=json.dumps(payload).encode();n=len(data);header=bytes([0x81,0x80|n]) if n<126 else bytes([0x81,0x80|126])+struct.pack('!H',n) if n<65536 else bytes([0x81,0x80|127])+struct.pack('!Q',n);mask=os.urandom(4);self.s.sendall(header+mask+bytes(v^mask[i%4] for i,v in enumerate(data)))
    def receive(self):
        data=b''
        while True:
            a,b=self.exact(2);n=b&127
            if n==126:n=struct.unpack('!H',self.exact(2))[0]
            if n==127:n=struct.unpack('!Q',self.exact(8))[0]
            mask=self.exact(4) if b&128 else None;payload=self.exact(n)
            if mask:payload=bytes(v^mask[i%4] for i,v in enumerate(payload))
            if (a&15)==8:raise EOFError('Chrome websocket closed')
            if (a&15) in [0,1]:data+=payload
            if a&128 and data:return json.loads(data)
    def call(self,name,params=None):
        self.n+=1;ident=self.n;self.send({'id':ident,'method':name,'params':params or {}})
        while True:
            r=self.receive()
            if r.get('id')==ident:
                if 'error' in r:raise RuntimeError(r['error'])
                return r.get('result',{})

def main():
    report=Path(sys.argv[1]).resolve();receipt=Path(sys.argv[2]);assert (report/'index.html').is_file();profile=BASE/('report_chrome_QA_'+str(int(time.time())));profile.mkdir();chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe');args=[str(chrome),'--headless=new','--no-first-run','--no-default-browser-check','--disable-background-networking','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+str(profile),'about:blank'];proc=subprocess.Popen(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW);cdp=None
    try:
        for _ in range(150):
            f=profile/'DevToolsActivePort'
            if f.exists():break
            if proc.poll() is not None:raise RuntimeError('Own Chrome exited before CDP ready')
            time.sleep(.1)
        else:raise TimeoutError('Own Chrome startup')
        port=int(f.read_text().splitlines()[0]);req=urllib.request.Request(f'http://127.0.0.1:{port}/json/new?about:blank',method='PUT');target=json.load(urllib.request.urlopen(req,timeout=10));cdp=CDP(target['webSocketDebuggerUrl']);cdp.call('Page.enable');cdp.call('Emulation.setDeviceMetricsOverride',{'width':1400,'height':1100,'deviceScaleFactor':1,'mobile':False});cdp.call('Page.navigate',{'url':(report/'index.html').as_uri()})
        expression="""(async()=>{await new Promise((resolve,reject)=>{let n=0;const t=setInterval(()=>{if(document.readyState==='complete'&&document.querySelector('h1')){clearInterval(t);resolve()}else if(++n>200){clearInterval(t);reject(Error('page timeout'))}},50)});const images=[...document.images];await Promise.all(images.map(i=>i.complete?Promise.resolve():new Promise((r,j)=>{i.onload=r;i.onerror=j})));const videos=[];for(const v of document.querySelectorAll('video')){v.muted=true;v.preload='auto';await v.play();await new Promise(r=>setTimeout(r,450));videos.push({src:v.getAttribute('src'),width:v.videoWidth,height:v.videoHeight,currentTime:v.currentTime,readyState:v.readyState,error:v.error&&v.error.code});v.pause()}return {title:document.querySelector('h1').textContent,images:images.length,allImagesLoaded:images.every(i=>i.naturalWidth>0&&i.naturalHeight>0),videos,links:[...document.querySelectorAll('a')].map(a=>a.getAttribute('href'))}})()"""
        result=cdp.call('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True});assert 'exceptionDetails' not in result,result;v=result['result']['value'];assert v['allImagesLoaded'] and v['images']>0 and all(x['width']==1920 and x['height']==1080 and x['currentTime']>.05 and x['error'] is None for x in v['videos']);assert all((report/link).is_file() for link in v['links']);screenshot=cdp.call('Page.captureScreenshot',{'format':'png'})['data'];qa=report/'QA';qa.mkdir(exist_ok=True);(qa/'chrome_report.png').write_bytes(base64.b64decode(screenshot));v.update(passed=True,report=str(report),separate_headless_Chrome_own_process_only=True,verified_local_links=True);receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(v,ensure_ascii=False),flush=True)
    finally:
        if cdp:cdp.s.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:raise RuntimeError('Own isolated Chrome did not terminate; preserve its identity/profile for followup')

if __name__=='__main__':main()
