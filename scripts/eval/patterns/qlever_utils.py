def build_qlever_cmd(
    qlever_cmd: list[str],
    *,
    singularity: bool = False,
    qlever_simg_filepath: str | None = None,
) -> list[str]:
    if not singularity:
        return qlever_cmd
    assert qlever_simg_filepath is not None
    cmd = ["singularity", "exec", f"{qlever_simg_filepath}"]
    cmd.extend(qlever_cmd)
    if qlever_cmd[:2] != ["qlever", "stop"]:
        cmd.extend(["--system", "native"])
    return cmd
