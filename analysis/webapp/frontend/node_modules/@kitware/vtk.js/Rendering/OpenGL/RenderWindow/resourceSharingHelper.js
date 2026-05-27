// See typescript header for documentation

function getTransferFunctionHash(transferFunction, useIndependentComponents, numberOfComponents) {
  return transferFunction ? `${transferFunction.getMTime()}-${useIndependentComponents}-${numberOfComponents}` : '0';
}
function getImageDataHash(image, scalars) {
  return `${image.getMTime()}A${scalars.getMTime()}`;
}
var resourceSharingHelper = {
  getTransferFunctionHash,
  getImageDataHash
};

export { resourceSharingHelper as default, getImageDataHash, getTransferFunctionHash };
