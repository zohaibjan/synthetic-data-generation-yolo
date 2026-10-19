import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).isoformat()

class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, created TEXT, status TEXT, request TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS samples(id TEXT PRIMARY KEY, job TEXT REFERENCES jobs(id), seed INTEGER, status TEXT, boxes TEXT, negative INTEGER DEFAULT 0, reviewed TEXT, sha256 TEXT, source_scene TEXT);
            ''')
    def db(self):
        db = sqlite3.connect(self.root / "state.db", timeout=30)
        db.row_factory = sqlite3.Row
        return db
    def submit(self, request):
        jid = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", (jid,now(),"queued",json.dumps(request.model_dump()),None))
        return jid
    def jobs(self):
        with self.db() as db:
            return [dict(r) | {"request":json.loads(r["request"])} for r in db.execute("SELECT * FROM jobs ORDER BY created DESC")]
    def job(self,jid):
        return next((j for j in self.jobs() if j["id"]==jid),None)
    def claim(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if not row: return None
            db.execute("UPDATE jobs SET status='running' WHERE id=?", (row[0],))
        return self.job(row[0])
    def status(self,jid,status,error=None):
        with self.db() as db:
            db.execute("UPDATE jobs SET status=?, error=? WHERE id=?", (status,error,jid))
    def recover(self):
        with self.db() as db:
            db.execute("UPDATE jobs SET status='interrupted',error='Worker stopped; submit a new job' WHERE status='running'")
    def samples(self,jid):
        with self.db() as db:
            return [dict(r) | {"boxes":json.loads(r["boxes"])} for r in db.execute("SELECT * FROM samples WHERE job=? ORDER BY seed",(jid,))]
    def sample(self,sid):
        with self.db() as db:
            row = db.execute("SELECT * FROM samples WHERE id=?",(sid,)).fetchone()
        return None if row is None else dict(row) | {"boxes":json.loads(row["boxes"])}
    def add_sample(self,jid,seed,boxes,digest,source_scene=None):
        sid = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?)", (sid,jid,seed,"pending",json.dumps(boxes),0,None,digest,source_scene))
        return sid
    def review(self,sid,review):
        with self.db() as db:
            db.execute("UPDATE samples SET status=?,boxes=?,negative=?,reviewed=? WHERE id=?",(review.status,json.dumps([b.model_dump() for b in review.boxes]),int(review.negative_confirmed),now(),sid))
    def image_path(self,sample):
        return self.root / "images" / sample["job"] / f'{sample["seed"]}.png'
