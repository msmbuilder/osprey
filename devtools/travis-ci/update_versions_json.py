import os
import json

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen
from osprey import version

if not version.release:
    print("This is not a release.")
    exit(0)

URL = 'http://www.msmbuilder.org/osprey'
res = urlopen(URL + '/versions.json')
versions = json.loads(res.read().decode('utf-8'))

# new release so all the others are now old
for i in range(len(versions)):
    versions[i]['latest'] = False

versions.append({
    'version': version.short_version,
    'url': "{base}/{version}".format(base=URL, version=version.short_version),
    'latest': True})

curpath = os.path.abspath(os.curdir)
savepath = os.path.join(curpath, "docs/_deploy/versions.json")
with open(savepath, 'w') as versionf:
    json.dump(versions, versionf)
