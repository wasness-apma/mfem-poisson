# MFEM Template

This is a cmake template for building MFEM applications.

## MFEM binding


```bash
cmake -DCMAKE_BUILD_TYPE=Release -DMFEM_DIR=<PATH_TO_MFEM_BUILD> -S . -B <PATH_TO_BUILD>
```
It is also possible to use in-source build 

## Change CMakeLists.txt

- `MyApp` -> Your application name

- `myapp` -> Your source name
    
