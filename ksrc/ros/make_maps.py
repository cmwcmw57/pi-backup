"""경사·단차 마스크를 Nav2 정적 맵으로 내보낸다 (ros/maps/).

경사 상한은 `config.max_slope_deg = 21도` 를 쓴다.  그 값은 로버 견인력 예산에서
유도된 것이다: 가용 마찰 0.53 - 굴림/침하 저항 0.14 = 0.39 -> atan = 21.3도.
이게 Nav2 가 **로버가 못 오르는 30도 램프를 피하게** 만드는 지점이다.
차체 크기는 nav2_params.yaml 의 footprint 와 inflation 에서 별도 적용한다.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sim" / "rl"))
import terrain as terr        # noqa: E402

for kind in ("sand", "rock"):
    y, inf = terr.export_costmap(kind, str(HERE / "maps"))
    print("%-5s -> %s   %dx%d  %.4f m/px  자유 %.1f%%  %.1f x %.1f m"
          % (kind, pathlib.Path(y).name, inf["nx"], inf["ny"], inf["res"],
             100 * inf["free"], inf["ex"], inf["ey"]))
