"""Render the NUC flight model from its original XML and content-addressed visual assets."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import mujoco

def make_plant():
    package=Path(os.environ.get('GATE_REGISTERED_CAD',str(Path(__file__).resolve().parents[1]/'records/registered-cad-assets.tgz')))
    with tarfile.open(package,'r:gz') as archive:
        source=archive.extractfile('source-model.xml').read()
        manifest=json.loads(archive.extractfile('asset-hashes.json').read())
        xml=ET.fromstring(source)
        assets={}
        for node in xml.findall('.//*[@file]'):
            original=node.attrib['file'];checksum,name=manifest[original]
            key=Path(name).name
            if key not in assets:
                content=archive.extractfile(name).read()
                if hashlib.sha256(content).hexdigest()!=checksum:raise IOError(name)
                assets[key]=content
            node.set('file',key)
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'),assets=assets)
    return SimpleNamespace(m=model,xml=source.decode(),xml_sha256=hashlib.sha256(source).hexdigest())
