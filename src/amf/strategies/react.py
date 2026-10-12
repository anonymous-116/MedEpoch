from __future__ import annotations

from amf.state import EpisodeState
from amf.strategies.base import BaseStrategy, RuntimeContext

class ReactStrategy(BaseStrategy):
    async def run(self, ctx: RuntimeContext, state: EpisodeState) -> None:
        for turn_id in range(ctx.max_turns):
            state.turn_id = turn_id
            await ctx.prepare_context(state)
            force_final = ctx.is_final_turn(turn_id)
            prompt = ctx.build_prompt(
                state,
                force_final=force_final,
            )
            decision = await ctx.agent_step(state, prompt, force_final=force_final)

            if decision.type == "final":
                ctx.finish(state, decision.answer or "")
                return

            if decision.type == "tool_call":
                if force_final:
                    ctx.record_error(
                        state,
                        "final answer turn returned a tool call",
                        raw_output=decision.raw_output,
                    )
                    return
                await ctx.call_tool(state, decision.tool_call or {})
                continue

            ctx.record_error(
                state,
                decision.error or "invalid model output",
                raw_output=decision.raw_output,
            )
            return

        state.set_status("MAX_TURNS")
        ctx.recorder.record(state.episode_id, state.turn_id, "error", {"error": "max turns reached"})
