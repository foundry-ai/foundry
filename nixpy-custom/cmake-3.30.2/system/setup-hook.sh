addCMakeParams() {
    addToSearchPath CMAKE_PREFIX_PATH $1
}

fixCmakeFiles() {
    # Replace occurences of /usr and /opt by /var/empty.
    echo "fixing cmake files..."
    find "$1" \( -type f -name "*.cmake" -o -name "*.cmake.in" -o -name CMakeLists.txt \) -print |
        while read fn; do
            sed -e 's^/usr\([ /]\|$\)^/var/empty\1^g' -e 's^/opt\([ /]\|$\)^/var/empty\1^g' < "$fn" > "$fn.tmp"
            mv "$fn.tmp" "$fn"
        done
}

addEnvHooks "$targetOffset" addCMakeParams

makeCmakeFindLibs(){
  isystem_seen=
  iframework_seen=
  for flag in ${NIX_CFLAGS_COMPILE-} ${NIX_LDFLAGS-}; do
    if test -n "$isystem_seen" && test -d "$flag"; then
      isystem_seen=
      export CMAKE_INCLUDE_PATH="${CMAKE_INCLUDE_PATH-}${CMAKE_INCLUDE_PATH:+:}${flag}"
    elif test -n "$iframework_seen" && test -d "$flag"; then
      iframework_seen=
      export CMAKE_FRAMEWORK_PATH="${CMAKE_FRAMEWORK_PATH-}${CMAKE_FRAMEWORK_PATH:+:}${flag}"
    else
      isystem_seen=
      iframework_seen=
      case $flag in
        -I*)
          export CMAKE_INCLUDE_PATH="${CMAKE_INCLUDE_PATH-}${CMAKE_INCLUDE_PATH:+:}${flag:2}"
          ;;
        -L*)
          export CMAKE_LIBRARY_PATH="${CMAKE_LIBRARY_PATH-}${CMAKE_LIBRARY_PATH:+:}${flag:2}"
          ;;
        -F*)
          export CMAKE_FRAMEWORK_PATH="${CMAKE_FRAMEWORK_PATH-}${CMAKE_FRAMEWORK_PATH:+:}${flag:2}"
          ;;
        -isystem)
          isystem_seen=1
          ;;
        -iframework)
          iframework_seen=1
          ;;
      esac
    fi
  done
}

# not using setupHook, because it could be a setupHook adding additional
# include flags to NIX_CFLAGS_COMPILE
postHooks+=(makeCmakeFindLibs)

