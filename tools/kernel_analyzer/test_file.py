import mujoco
import warp as wp
from packaging import version

from . import math
from . import support
from .types import MJ_MINVAL
from .types import CamLightType
from .types import Data
from .types import DisableBit
from .types import JointType
from .types import Model
from .types import TrnType
from .types import WrapType
from .types import array2df
from .types import array3df
from .types import vec10
from .warp_util import event_scope
from .warp_util import kernel
from .warp_util import kernel_copy


def test_function(m: Model, d: Data):
  @kernel
  def _root(m: Model, d: Data):
    worldid = wp.tid()
    d.xpos[worldid, 0] = wp.vec3(0.0)


@kernel
def _root(m: Model, d: Data, hi: int = 1, *args, **kwargs):
  worldid = wp.tid()
  d.xpos[worldid, 0] = wp.vec3(0.0)


@kernel
def test_model_data_in_the_middle(
  qpos: wp.array(float, ndim=1),
  random: int,
  qvel_in: wp.array(float, ndim=1),
  xpos_out: wp.array(float, ndim=1),
  qpos0: wp.array(float, ndim=1),
  random2: int,
  qpos0_in: wp.array(float, ndim=1),
  qpos0_out: wp.array(float, ndim=1),
  xpos_outt: wp.array(float, ndim=1),
):
  worldid = wp.tid()


@kernel
def test_model_data_in_the_middle(
  xpos_out: wp.array(float, ndim=1),
  qvel_in: wp.array(float, ndim=1),
  qpos: wp.array(float, ndim=1),
):
  worldid = wp.tid()



@kernel
def test_comments(
  # Model
  qpos0: wp.array(float, ndim=1),
  # Data
  qvel: wp.array(float, ndim=1),
  # Data in
  qvel_in: wp.array(float, ndim=1),
  # Data out
  xpos_out: wp.array(float, ndim=1),
):
  worldid = wp.tid()
