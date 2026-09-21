# igris_c_gmr

VR 헤드셋(vr_system, VRS1) 텔레오퍼레이션 **프로듀서**. 사람 스켈레톤을 GMR로 IGRIS-C에
리타게팅하여 `igris_c_inference_node`가 구독하는 ROS 2 모션 레퍼런스 토픽을 직접 발행하고,
손가락 명령과 컨트롤러 버튼(제어모드 전환)은 igris_c_sdk(CycloneDDS)로 로봇에 직접 보낸다.
hit-bench의 `scripts/teleoperation/vr_system_teleop.py`를 이전한 것으로, 중간의 Redis와 gmr
`redis_to_ros2_bridge.py` 단계는 없다.

> DRAFT. ROBROS AI 거버넌스: 이 프로듀서의 DDS 출력은 실제 로봇을 움직이고 제어모드를 바꾼다.
> 로봇에 연결하기 전 하드웨어 변경 승인과 아래 검증 절차의 사람 수행 기록이 필요하다.

## 데이터 흐름

```
headset (vr_system, VRS1 FlatBuffers over TCP -> this host :9102)
  -> vr_link.TcpVRDevice (gmr joint convention)
  -> GeneralMotionRetargeting(xrobot -> robros_igris_c_v2)          38 qpos = root 7 + 31 hinges
  -> NeckOverride                                                   HMD-relative neck yaw/pitch into qpos
  -> RetargetFramePayloadBuilder -> to_ros_payload (JSON)
  -> ROS 2 std_msgs/String  /gmr/teleop/retarget_frame  (domain 91, RELIABLE depth 1)
       -> igris_c_inference_node  motion term (23 policy joints -> ONNX WBC -> DDS lowcmd)
                                  motion_command term (6 wrist + 2 neck -> direct joint targets)
  -> HandTargets -> HandCmdDdsPublisher     DDS <ns>/handcmd  (12 normalized motors, 100 Hz)
  -> ControllerEvents -> ControlModeClient  DDS <ns>/service/control_mode/request
```

ROS 토픽에는 **전신 모션 프레임만** 실린다. 키: `fps, root_pos, root_rot(wxyz), root_vel,
root_angvel, dof_pos, dof_vel (MJCF 순서 31), keybody_pos_local, keybody_rot_local,
link_body_list, record_meta`. 소비자 파서(igris_c_inference_public `MotionReferenceSource.cpp`)의
필수 키는 `dof_pos, dof_vel, root_rot, link_body_list, keybody_rot_local`이며 나머지는 무시된다.
`tests/consumer_oracle.py`가 그 파서를 파이썬으로 옮겨 계약 테스트에 쓴다.

컨트롤러 버튼 매핑(상승 엣지 1회, 스켈레톤 연결 중일 때만; 퇴역한 `igris_c_gmt_public`의
`gmr_control_mode_dds_bridge`와 동일):

| 버튼 | 이벤트 | DDS 제어모드 요청 |
|---|---|---|
| 오른손 B | `lowcmd_start` | `CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL` (inference 노드가 스스로 진입하므로 중복) |
| 오른손 썸스틱 클릭 | `stop` | `CONTROL_MODE_CMD_JOINT_POSITION_HOLD` |
| 왼손 썸스틱 클릭 | `go_home` | `CONTROL_MODE_CMD_MOTION_PRESET`, preset `HOME` |

오른손 A는 헤드셋 상태머신이 소비한다. 헤드셋이 Paused면 고정 스냅샷을 재전송하므로 버튼은
Active 상태에서만 감지된다. 서비스 호출은 워커 스레드에서 블로킹되며(기본 3 s 타임아웃) 100 Hz
루프를 막지 않는다. 큐(4)가 가득 차면 새 요청은 버리고 경고를 남긴다.

## 레이아웃

```
pyproject.toml              pixi 환경(RoboStack ROS 포함) + 태스크 + 패키지 메타데이터 + pytest 설정
scripts/activate_ros.sh     pixi 활성화 훅: PYTHONPATH=thirdparty/gmr 만 (ROS·source·vr_link는 환경에 설치됨)
scripts/install_sdk_wheel.sh
source/igris_c_gmr/         editable 설치되는 패키지; 콘솔 명령 `igris-c-gmr` (= python -m igris_c_gmr)
  producer.py               진입점(argparse, 100 Hz 루프, init/teardown 순서)
  vrs1_source.py            TcpVRDevice 래퍼, .vrs1 리플레이, GMR 입력 변환
  neck.py                   HMD-상대 목 각도, MJCF 범위 클램프
  hands.py                  12모터 정규화 손 명령 (gmr igris_c_hand_command)
  controller.py             버튼 엣지 검출, 이벤트 -> 제어모드 테이블
  retarget_frame.py         FK keybody + 속도(유한차분, LPF) -> 계약 프레임, JSON 직렬화
  ros_publisher.py          rclpy String 퍼블리셔 (spin 없음)
  dds_io.py                 igris_c_sdk 채널, HandCmd 퍼블리셔, 제어모드 서비스 클라이언트(워커)
tests/                      pytest; consumer_oracle.py = C++ 파서 포트
thirdparty/{gmr, vr_link_schema, igris_c_sdk}   서브모듈
```

## 환경 (pixi)

한 프로세스에 GMR(mujoco/mink) + rclpy + igris_c_sdk가 함께 필요하다. 환경은 `.pixi/envs/default`
하나에 전부 들어가며 호스트의 `/opt/ros`는 쓰지 않는다 (2026-09-21 전환).

- **ROS 2 Jazzy**는 RoboStack(`robostack-jazzy`) conda 패키지다. 채널 순서와 버전 범위는
  로봇 NUC가 쓰는 `~/workspace/repos/igris_c` `pyproject.toml`의 `core` feature와 같게 맞췄다
  (`rclpy >=7.1.9,<8`, `std-msgs >=5.3.6,<6`, `rmw-fastrtps-cpp >=8.4.3,<9`). RoboStack `.so`는
  `RPATH`를 쓰므로 로그인 셸의 `LD_LIBRARY_PATH`(igris_c install, /opt/ros)가 상속되어도 환경 안의
  `librcl.so`가 우선한다(ldd로 확인).
- **이 레포**(`source/igris_c_gmr`)와 **vr_link_schema**는 각자의 pyproject.toml로 editable 설치된다.
  `igris-c-gmr` 콘솔 명령이 환경 `bin/`에 생긴다.
- **gmr**만 활성화 훅이 `PYTHONPATH`로 노출한다(setup.py가 smplx/opencv/redis를 강제하므로).
  `PYTHONPATH`는 처음부터 다시 쓰므로 로그인 셸의 `igris_c/install` rosidl `igris_c_sdk`가
  pybind 바인딩을 가리는 일은 없다.
- **igris_c_sdk 휠**은 사람이 빌드해 pip으로 넣는다. 그 `.so`는 호스트 libssl/libstdc++에 링크되어
  있지만, 프로듀서는 ROS를 먼저 초기화하므로 실제 프로세스에는 환경의 libssl/libstdc++가 먼저 올라오고
  SDK는 그것을 재사용한다(`/proc/self/maps`로 확인).

```bash
git submodule update --init --recursive
pixi install

# igris_c_sdk 파이썬 바인딩 휠 (Python 3.12, 사람이 빌드)
(cd thirdparty/igris_c_sdk && task wheel)     # -> dist/igris_c_sdk-*.whl
pixi run install-sdk

pixi run check-env      # igris_c_sdk·rclpy 경로가 모두 .pixi/envs 아래이고 True 출력
pixi run test           # 단위/계약 테스트
pixi run test-ros       # 실 rclpy 라운드트립
```

`pixi run --clean-env ...`는 호스트 환경변수를 모두 버리지만 `bash`, `ldd` 같은 호스트 명령도 함께
사라지므로 파이썬 태스크(`teleop`, `replay`)에만 쓴다.

**셋업 트러블슈팅 (2026-09-18 첫 설치에서 확인)**

- `task wheel`이 `No download info given for 'cyclonedds'`로 실패: igris_c_sdk의 중첩 서브모듈
  (cyclonedds, cyclonedds-cxx, pybind11)이 비어 있다. 루트에서 `git submodule update --init --recursive`.
- `pixi run check-env`가 `No module named 'imageio'`: gmr 패키지 import가 뷰어 모듈을 거친다.
  `pyproject.toml`의 pypi-dependencies에 포함되어 있으므로 `pixi install`을 다시 실행.
- pytest가 `launch_testing` 플러그인 오류로 죽으면 ROS pytest 플러그인 자동 로드 때문이다.
  `pixi run test` / `test-ros` 태스크는 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`을 이미 설정한다.
- `pixi install`을 다시 돌려도 pip으로 넣은 SDK 휠은 유지된다(RoboStack 전환으로 환경을 재해결한
  2026-09-21에도 유지됨).
- editable 설치가 `source/igris_c_gmr.egg-info/`를 만든다. `.gitignore`에 있다.

**바인딩 함정**: 옛 빌드의 `.so`는 DDS 타입 이름을 `igris_c_sdk::msg::dds_::HandCmd_`로 등록해
로봇 브리지(`igris_c::msg::dds::HandCmd`)와 조용히 불일치했다. 항상 서브모듈 핀에서 새로 빌드해
설치하고, `strings <so> | grep -o 'igris_c.*::HandCmd'`에 `dds_`가 없는지 확인한다.
`pixi run check-env`가 "SHADOWED by rosidl package"를 찍으면 PYTHONPATH에 igris_c 오버레이가 남아 있다.

**서비스 토픽 접두 함정(2026-09-18 확인)**: `igris_c_teleop_env`에 남아 있던 2026-09-15 휠(igris_c 2ba902e
빌드)의 `IgrisC_Client`는 `rt/service/control_mode/request`에 요청을 쓴다. 서브모듈 핀(6b40103)과 현
igris_c 브리지(`dds_manager.cpp`)는 `service/control_mode/request`다. 옛 휠이면 버튼 요청이 브리지에
도달하지 않으며 실패도 조용하다(타임아웃만). 프로듀서는 시작 시 로드한 SDK 버전·경로를 로그로 남기니
서브모듈에서 빌드한 휠인지 확인한다. `HandCmd`의 `handcmd` 토픽은 두 버전에서 같다.

## 실행 순서 (로봇 NUC)

1. igris_c 컨트롤러/브리지 (DDS `lowstate`, `lowcmd`, `handcmd`, 제어모드 서비스) - 사람이 기동.
2. 프로듀서:
   ```bash
   pixi run teleop -- --dds_namespace igris_c_<unit> --measure_fps
   ```
   헤드셋을 Active로 두고 로그에 `VR skeleton connected`가 뜨는지 확인한다.
3. `ros2 run igris_c_inference_public igris_c_inference_node --config <policy>/params.yaml`
   (command.yaml의 `source.params.domain_id`가 91, 토픽이 `/gmr/teleop/retarget_frame`).

**순서가 중요하다.** inference 노드는 시작 후 5 s 안에 46 프레임을 받지 못하면 종료한다. 이
프로듀서는 스켈레톤이 없으면 아무것도 발행하지 않으므로(기존 브리지의 기본 자세 시딩 없음)
2번이 스트리밍 중일 때 3번을 띄운다. 운용 중 끊김은 inference의 `on_stale: hold_last`가 마지막
레퍼런스를 유지하고, 재접속 시 재개된다.

주요 옵션: `--no_dds`(ROS만, 손/버튼 미전송), `--replay_vrs1 <file>`(헤드셋 없이 녹화 재생),
`--finger_source auto|hand|grip|none`, `--dds_topic_naming native|ros_compatible`(로봇
`igris_c.network.dds_topic_naming`과 일치), `--hand_dds_topic handcmd`, `--no_mode_events`,
`--ros_domain_id 91`, `--invert_neck_yaw/--invert_neck_pitch`.

DDS 네임스페이스는 기본값이 없다. 틀린 네임스페이스는 오류 없이 아무 데도 도달하지 않으므로 시작
로그의 resolve된 토픽명(`<ns>/handcmd`, `<ns>/service/control_mode/request`)을 로봇 브리지가
구독하는 이름과 대조한다. 현재 igris_c 브리지 소스(`igris_c_bridge/src/dds/dds_manager.cpp`)는
native 네이밍의 bare `handcmd`를 구독한다. 로봇 NUC의 브리지 버전이 다르면(`rt/` 접두 구버전)
`--hand_dds_topic rt/handcmd`로 맞춘다.

## 검증 절차 (사람 수행, 합격 기준)

1. `pixi run check-env`: igris_c_sdk 경로가 `.pixi/envs` 아래, `True` 출력.
2. `pixi run test` 전부 통과, `pixi run test-ros` 통과.
3. 헤드셋 없이: `pixi run replay -- <file>.vrs1`. 다른 ROS 셸에서
   `ROS_DOMAIN_ID=91 ros2 topic hz /gmr/teleop/retarget_frame` ≈ 100 Hz,
   `ros2 topic echo --once`에 `dof_pos` 31개·`link_body_list` 존재,
   `ros2 topic info -v`에 RELIABLE / KEEP_LAST(1).
4. inference 노드(토크 OFF 또는 시뮬)를 리플레이에 붙여 5 s 내 준비 로그, `invalid motion-reference payload` 없음.
   프로듀서 종료 → 소비자 hold, 재시작 → 재개.
5. 헤드셋 실연결로 3·4 반복. 끊김 시 발행 정지, 재접속 시 재개.
6. DDS 손 경로(토크 OFF, 승인 기록): 시작 로그의 resolve 토픽명이 브리지 구독명과 일치.
   SDK `HandState`/`handcmd` 구독자로 12모터 id 11-16/21-26, q∈[0,1], 100 Hz, 스켈레톤 없을 때 0 Hz.
7. 제어모드(승인 필요, 안전 자세): 오른 B / 오른 스틱 / 왼 스틱 각 1회 → 로그에 요청 1건씩과
   `accepted`; 로봇 모드 전환 확인; `--measure_fps`로 요청 중 루프 주기 유지. 누른 채 유지 시 재발화 없음.
8. Ctrl-C: 깨끗한 종료, DDS/rmw abort 없음, exit 0.

롤백: hit-bench의 Redis + `run_ros2_redis_bridge.sh` 경로는 그대로 남아 있다.

## 미해결 (사람 확인)

- 로봇 NUC의 igris_c 브리지/SDK 버전과 토픽 네이밍. 서브모듈 핀 `igris_c_sdk@6b40103`과
  `igris_c` 워크스페이스 `78bec2c`의 와이어는 동일(차이는 테스트 스냅샷 1개).
- `pixi install`이 pip으로 넣은 SDK 휠을 유지하는지 최초 1회 확인.
- 손가락 정규화 → 관절 각도 매핑은 선형 플레이스홀더(gmr `igris_c_hand_command`); 하드웨어 굽힘 비율 측정 전.
