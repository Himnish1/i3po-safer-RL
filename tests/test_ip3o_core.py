import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    F = None

if torch is not None:
    from ip3o.algorithms.ip3o import IP3O, IP3OConfig
    from ip3o.algorithms.ppo_lag import PPOLagConfig, PPOLagrangian
    from ip3o.models.actor_critic import ActorCritic
    from ip3o.utils.buffer import RolloutBuffer


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class TestIP3OCore(unittest.TestCase):
    def test_dual_critic_shapes(self):
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        obs = torch.zeros((3, 4), dtype=torch.float32)
        action, logp, v_r, v_c = model.act(obs)

        self.assertEqual(action.shape, (3, 2))
        self.assertEqual(logp.shape, (3,))
        self.assertEqual(v_r.shape, (3,))
        self.assertEqual(v_c.shape, (3,))

    def test_incremental_penalty_increases_as_cost_approaches_limit(self):
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        cfg = IP3OConfig(cost_limit=10.0, eta=1.0, gamma=0.99)
        algo = IP3O(model, cfg, device=torch.device("cpu"))

        # L_C is small when cost is well below limit
        low_l_cost = algo.cfg.eta * F.celu(torch.tensor(-5.0))
        # L_C grows as cost approaches and exceeds limit
        high_l_cost = algo.cfg.eta * F.celu(torch.tensor(8.0))

        self.assertLess(low_l_cost.item(), high_l_cost.item())

    def test_celu_penalty_self_scales_with_l_cost(self):
        """Test that CELU barrier self-scales: eta * CELU(L_C) grows as constraint is violated."""
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(16, 16))
        cfg = IP3OConfig(eta=1.0, cost_limit=10.0, gamma=0.99)
        algo = IP3O(model, cfg, device=torch.device("cpu"))

        # When L_C < 0 (safe), CELU output is small negative
        safe_penalty = algo.cfg.eta * F.celu(torch.tensor(-2.0))
        # When L_C > 0 (violated), CELU output is linear and grows
        violated_penalty = algo.cfg.eta * F.celu(torch.tensor(5.0))

        self.assertLess(safe_penalty.item(), 0.0)
        self.assertGreater(violated_penalty.item(), 0.0)
        self.assertGreater(abs(violated_penalty.item()), abs(safe_penalty.item()))

    def test_cost_returns_are_discounted_cost_to_go(self):
        buffer = RolloutBuffer(gamma=0.9, gae_lambda=0.95)
        buffer.add([0.0], [0.0], 0.0, 1.0, False, 0.0, 0.0, 0.0)
        buffer.add([0.0], [0.0], 0.0, 2.0, False, 0.0, 0.0, 0.0)
        batch = buffer.get(torch.device("cpu"), last_vr=0.0, last_vc=0.5)

        expected = torch.tensor([3.09475, 2.45], dtype=torch.float32)
        self.assertTrue(torch.allclose(batch["cost_returns"], expected, atol=1e-6))
        self.assertTrue(torch.all(batch["cost_returns"] >= 0))

    def test_low_kl_does_not_stop_inner_loop_by_default(self):
        class DummyAC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

        class DummyPPO(PPOLagrangian):
            def __init__(self, actor_critic, config, device):
                super().__init__(actor_critic, config, device)
                self.policy_calls = 0
                self.value_calls = 0

            def _policy_loss(self, batch):
                self.policy_calls += 1
                loss = self.ac.weight.pow(2)
                approx_kl = torch.tensor(0.0005, device=self.device)
                return loss, approx_kl

            def _value_loss(self, batch):
                self.value_calls += 1
                return self.ac.weight.pow(2)

        model = DummyAC()
        cfg = PPOLagConfig(train_iters=3, target_kl=0.015, kl_lower=0.0, policy_lr=0.1)
        agent = DummyPPO(model, cfg, device=torch.device("cpu"))
        batch = {"cost_returns": torch.tensor([0.0], dtype=torch.float32)}

        info = agent.update(batch)

        self.assertEqual(agent.value_calls, 3)
        self.assertEqual(agent.policy_calls, 3)
        self.assertAlmostEqual(info["kl"], 0.0005, places=6)


if __name__ == "__main__":
    unittest.main()
