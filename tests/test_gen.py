"""Generator backend tests (no codex needed): python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from texforge import gen  # noqa: E402


class Backends(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        gen.set_backend(None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        gen.set_backend(None)

    def test_default_is_codex(self):
        os.environ.pop("TEXFORGE_GENERATOR", None)
        self.assertEqual(gen.backend(), "codex")

    def test_none_raises_clear_error(self):
        os.environ["TEXFORGE_GENERATOR"] = "none"
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "pass an existing image"):
                gen.generate("p", Path(d), "x")
            with self.assertRaisesRegex(RuntimeError, "pass an existing image"):
                gen.generate_edit("p", Path(d) / "in.png", Path(d), "x")

    def test_option_overrides_env(self):
        os.environ["TEXFORGE_GENERATOR"] = "command"
        gen.set_backend("none")
        self.assertEqual(gen.backend(), "none")
        with self.assertRaises(ValueError):
            gen.set_backend("bogus")

    def test_command_template(self):
        os.environ["TEXFORGE_GENERATOR"] = "command"
        os.environ["TEXFORGE_GENERATOR_COMMAND"] = (
            f'"{sys.executable}" -c "import sys,shutil; open(sys.argv[1],\'w\').write(sys.argv[2])" {{output}} {{prompt}} {{input}}')
        with tempfile.TemporaryDirectory() as d:
            files = gen.generate("hello prompt", Path(d), "m", n=2, parallel=1)
            self.assertEqual(len(files), 2)
            self.assertEqual(files[0].read_text(), "hello prompt")

    def test_command_missing_output_fails(self):
        os.environ["TEXFORGE_GENERATOR"] = "command"
        os.environ["TEXFORGE_GENERATOR_COMMAND"] = f'"{sys.executable}" -c "pass"'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                gen.generate("p", Path(d), "m")


if __name__ == "__main__":
    unittest.main()
