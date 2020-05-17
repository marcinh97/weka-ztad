package lab2;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

@AllArgsConstructor
@Getter
@ToString
public class ClassificationItem<T> {
    private T value;
    private String givenClass;
}
