# Submodules
git_include(Eigen)

add_library(core STATIC
    ${CMAKE_SOURCE_DIR}/src/classifier.h
    ${CMAKE_SOURCE_DIR}/src/classifier.cpp
    )

target_include_directories(core
    PUBLIC ${CMAKE_SOURCE_DIR}/src
    )

target_link_libraries(core PUBLIC
    Eigen
    )

if (DO_CLANG_TIDY)
    set_target_properties(core PROPERTIES CXX_CLANG_TIDY "${DO_CLANG_TIDY}")
endif ()