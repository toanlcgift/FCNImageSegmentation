using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp
{
    public class DynamicBackend : Backend
    {
        private string _backend;

        public DynamicBackend(string backend = null)
        {
            _backend = backend ?? GetDefaultBackend();
        }

        public void SetBackend(string backend)
        {
            var availableBackends = new HashSet<string> { "tensorflow", "jax", "torch", "numpy", "openvino" };
            if (!availableBackends.Contains(backend))
            {
                throw new ArgumentException($"Available backends are ('tensorflow', 'jax', 'torch', 'numpy', and 'openvino'). Received: backend={backend}");
            }
            _backend = backend;
        }

        public void Reset()
        {
            _backend = GetDefaultBackend();
        }

        public string Name => _backend;

        public object GetAttribute(string name)
        {
            switch (_backend)
            {
                case "tensorflow":
                    return GetBackendAttribute("Keras.Backend.Tensorflow", name);
                case "jax":
                    return GetBackendAttribute("Keras.Backend.Jax", name);
                case "torch":
                    return GetBackendAttribute("Keras.Backend.Torch", name);
                case "numpy":
                    if (GetDefaultBackend() == "numpy")
                    {
                        return GetBackendAttribute("Keras.Backend", name);
                    }
                    else
                    {
                        throw new NotImplementedException("Currently, we cannot dynamically import the numpy backend because it would disrupt the namespace of the import.");
                    }
                case "openvino":
                    return GetBackendAttribute("Keras.Backend.Openvino", name);
                default:
                    throw new InvalidOperationException("Unsupported backend.");
            }
        }

        private string GetDefaultBackend()
        {
            // Replace this with the logic to get the default backend.  
            return "tensorflow";
        }

        private object GetBackendAttribute(string moduleName, string attributeName)
        {
            var module = Assembly.Load(moduleName);
            var type = module.GetType(moduleName);
            return type?.GetProperty(attributeName)?.GetValue(null);
        }
    }
}
