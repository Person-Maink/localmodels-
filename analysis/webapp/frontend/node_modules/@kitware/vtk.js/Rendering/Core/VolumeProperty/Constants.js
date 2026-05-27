const InterpolationType = {
  NEAREST: 0,
  LINEAR: 1,
  FAST_LINEAR: 2
};
const OpacityMode = {
  FRACTIONAL: 0,
  PROPORTIONAL: 1
};
const ColorMixPreset = {
  CUSTOM: 0,
  ADDITIVE: 1,
  COLORIZE: 2
};
var Constants = {
  InterpolationType,
  OpacityMode,
  ColorMixPreset
};

export { ColorMixPreset, InterpolationType, OpacityMode, Constants as default };
