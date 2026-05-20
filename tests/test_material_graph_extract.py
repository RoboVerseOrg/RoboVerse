from roboverse_pack.blender.usd.material_graph.extract import source_asset_from_shader


class _Value:
    def __init__(self, value):
        self._value = value

    def Get(self):
        return self._value


class _Asset:
    def __init__(self, path):
        self.path = path


class _Prim:
    def __init__(self, value=None):
        self._value = value

    def GetAttribute(self, name):
        if name == "info:mdl:sourceAsset" and self._value is not None:
            return _Value(self._value)
        return None


class _Shader:
    def __init__(self, source_asset=None, info_asset=None, input_asset=None):
        self._source_asset = source_asset
        self._prim = _Prim(info_asset)
        self._input_asset = input_asset

    def GetSourceAsset(self, source_type):
        assert source_type == "mdl"
        return self._source_asset

    def GetPrim(self):
        return self._prim

    def GetInput(self, name):
        if name == "mdl:sourceAsset" and self._input_asset is not None:
            return _Value(self._input_asset)
        return None


def test_source_asset_from_shader_prefers_get_source_asset():
    shader = _Shader(
        source_asset=_Asset("from_source_asset.mdl"),
        info_asset=_Asset("from_info_attr.mdl"),
        input_asset=_Asset("from_input.mdl"),
    )

    assert source_asset_from_shader(shader) == "from_source_asset.mdl"


def test_source_asset_from_shader_reads_info_attribute_before_legacy_input():
    shader = _Shader(info_asset=_Asset("from_info_attr.mdl"), input_asset=_Asset("from_input.mdl"))

    assert source_asset_from_shader(shader) == "from_info_attr.mdl"


def test_source_asset_from_shader_falls_back_to_legacy_input():
    shader = _Shader(input_asset=_Asset("from_input.mdl"))

    assert source_asset_from_shader(shader) == "from_input.mdl"


def test_source_asset_from_shader_ignores_empty_get_source_asset():
    shader = _Shader(source_asset=_Asset(""), info_asset=_Asset("from_info_attr.mdl"))

    assert source_asset_from_shader(shader) == "from_info_attr.mdl"


def test_source_asset_from_shader_ignores_blank_info_attribute():
    shader = _Shader(info_asset=_Asset("  "), input_asset=_Asset("from_input.mdl"))

    assert source_asset_from_shader(shader) == "from_input.mdl"
