package net.nora.register;

import java.util.HashMap;
import java.util.Map;

public class Util {

    public static <T, E> Map<E, T> reverseMap(Map<T, E> map){
        final Map<E, T> reversedMap = new HashMap<>();
        for(Map.Entry<T, E> entry : map.entrySet()){
            reversedMap.put(entry.getValue(), entry.getKey());
        }
        return reversedMap;
    }

}
