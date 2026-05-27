import { m as macro } from '../../macros2.js';
import vtkImplicitFunction from './ImplicitFunction.js';

// ----------------------------------------------------------------------------
// Global methods
// ----------------------------------------------------------------------------

function evaluate(radius, center, xyz) {
  if (!Array.isArray(radius)) {
    const retVal = (xyz[0] - center[0]) * (xyz[0] - center[0]) + (xyz[1] - center[1]) * (xyz[1] - center[1]) + (xyz[2] - center[2]) * (xyz[2] - center[2]) - radius * radius;
    return retVal;
  }
  const r = [(xyz[0] - center[0]) / radius[0], (xyz[1] - center[1]) / radius[1], (xyz[2] - center[2]) / radius[2]];
  return r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - 1;
}

// ----------------------------------------------------------------------------
// Static API
// ----------------------------------------------------------------------------

const STATIC = {
  evaluate
};

// ----------------------------------------------------------------------------
// vtkSphere methods
// ----------------------------------------------------------------------------

function vtkSphere(publicAPI, model) {
  // Set our className
  model.classHierarchy.push('vtkSphere');
  publicAPI.evaluateFunction = xyz => evaluate(model.radius, model.center, xyz);
  publicAPI.evaluateGradient = xyz => {
    const retVal = [2.0 - (xyz[0] - model.center[0]), 2.0 - (xyz[1] - model.center[1]), 2.0 - (xyz[2] - model.center[2])];
    return retVal;
  };
}

// ----------------------------------------------------------------------------
// Object factory
// ----------------------------------------------------------------------------

const DEFAULT_VALUES = {
  radius: 0.5,
  center: [0.0, 0.0, 0.0]
};

// ----------------------------------------------------------------------------

function extend(publicAPI, model) {
  let initialValues = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
  Object.assign(model, DEFAULT_VALUES, initialValues);

  // Object methods
  vtkImplicitFunction.extend(publicAPI, model, initialValues);
  macro.setGet(publicAPI, model, ['radius']);
  macro.setGetArray(publicAPI, model, ['center'], 3);
  vtkSphere(publicAPI, model);
}

// ----------------------------------------------------------------------------

const newInstance = macro.newInstance(extend, 'vtkSphere');

// ----------------------------------------------------------------------------

var vtkSphere$1 = {
  newInstance,
  extend,
  ...STATIC
};

export { STATIC, vtkSphere$1 as default, extend, newInstance };
