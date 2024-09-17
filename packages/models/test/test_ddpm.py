import foundry.numpy as jnp
import foundry.random
from foundry.diffusion.ddpm import DDPMSchedule

def test_schedule_mode():
    schedule = DDPMSchedule.make_squaredcos_cap_v2(100, prediction_type="epsilon")
    sample = foundry.random.uniform(foundry.random.key(40), (32, 32, 3))

    noised, _, target = schedule.add_noise(foundry.random.key(42), sample, 50)
    test_target = schedule.output_from_denoised(noised, 50, sample)
    norm = jnp.linalg.norm(target - test_target)
    assert norm < 1e-4