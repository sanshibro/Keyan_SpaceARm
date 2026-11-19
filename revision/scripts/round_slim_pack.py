"""Self-contained material paths and estimator output, split for the meeting archive."""
from round_slim_box import OUT
from round_wrist_pack import pack,sha
from round_wrist_fusion import savej
def main():
    dest=OUT/'evidence';dest.mkdir(exist_ok=True)
    plant=list((OUT/'kinematics').glob('*.npz'))+[OUT/'sweep.npz',OUT/'sweep.json']
    bench=[p for q in ('p00','p15','p30','p40') for p in (OUT/q).rglob('*') if p.is_file()]
    bench += [OUT/n for n in ('box-profile.json','stationary-box.npz','bench-scores.json')]
    flight=[p for q in ('primary','natural_mount','mount_35mm','return_offset','gain_half','gain_double') for p in (OUT/q).rglob('*.npz')]
    flight += [OUT/'flight-scores.json',OUT/'audit.json']
    catalog={}
    for name,paths in [('slim-material-paths.tar.gz',plant),('slim-bench.tar.gz',bench),('slim-flights.tar.gz',flight)]:
        path=dest/name;pack(path,{f.relative_to(OUT).as_posix():f for f in paths})
        assert 0<path.stat().st_size<24*1024**2
        catalog[name]=dict(bytes=path.stat().st_size,sha256=sha(path),members=len(paths))
    savej(OUT/'evidence-index.json',catalog); print(catalog,flush=True)
if __name__=='__main__':main()
