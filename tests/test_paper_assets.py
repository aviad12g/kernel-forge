from scripts.build_paper_assets import _latex_escape


def test_latex_escape_does_not_reescape_generated_commands():
    assert _latex_escape(r"path\value_{x}") == (
        r"path\textbackslash{}value\_\{x\}"
    )
